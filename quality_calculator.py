#!/usr/bin/env python3
# quality_calculator.py

import os
import json
import h5py
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from astropy.convolution import convolve_fft, Gaussian1DKernel
from scipy.interpolate import interp1d

logger = logging.getLogger("QualityCalculator")

class QualityCalculator:
    """
    Calculate quality values for PHOENIX NLTE stellar models by comparing 
    synthetic spectra to observed spectra.
    """
    
    def __init__(self, nlte_models_dir: str, quality_output_dir: str, 
                 original_spectrum_file: str, recalculate_all: bool = False,
                 threads: int = 4):
        """
        Initialize the QualityCalculator.
        
        Args:
            nlte_models_dir: Directory containing NLTE model files (H5 format)
            quality_output_dir: Directory for storing quality calculation results
            original_spectrum_file: Path to the original observed spectrum file
            recalculate_all: Whether to recalculate all quality values or only missing ones
            threads: Number of concurrent threads to use for calculation
        """
        self.nlte_models_dir = nlte_models_dir
        self.quality_output_dir = quality_output_dir
        self.original_spectrum_file = original_spectrum_file
        self.recalculate_all = recalculate_all
        self.threads = threads
        
        # Ensure output directory exists
        os.makedirs(self.quality_output_dir, exist_ok=True)
        
        # Load original spectrum
        try:
            self.org_data = pd.read_csv(self.original_spectrum_file)
            self.org_wl = np.array(self.org_data['WL'].to_list())
            self.org_flux = np.array(self.org_data['Flux'].to_list())
            logger.info(f"Loaded original spectrum with {len(self.org_wl)} points")
        except Exception as e:
            logger.error(f"Failed to load original spectrum: {e}")
            raise

    def run(self) -> Optional[str]:
        """
        Run quality calculations for all NLTE models.
        
        Returns:
            Path to the output JSON file containing quality results, or None if failed
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = os.path.join(self.quality_output_dir, f"quality_results_{timestamp}.json")
        
        # Initialize special points list
        self.special_points = []
        
        # Check if we should load existing results
        existing_results = {}
        if not self.recalculate_all:
            # Find most recent quality results file
            existing_files = [f for f in os.listdir(self.quality_output_dir) 
                            if f.startswith("quality_results_") and f.endswith(".json")]
            if existing_files:
                existing_files.sort(reverse=True)  # Sort by name (timestamp)
                latest_file = os.path.join(self.quality_output_dir, existing_files[0])
                try:
                    with open(latest_file, 'r') as f:
                        existing_results = json.load(f)
                    logger.info(f"Loaded existing quality results from {latest_file}")
                except Exception as e:
                    logger.warning(f"Failed to load existing results from {latest_file}: {e}")
        
        # Get list of NLTE model files
        nlte_files = [os.path.join(self.nlte_models_dir, f) for f in os.listdir(self.nlte_models_dir) 
                    if f.endswith('.h5') and 'nlte' in f]
        
        if not nlte_files:
            logger.error(f"No NLTE model files found in {self.nlte_models_dir}")
            return None
        
        logger.info(f"Found {len(nlte_files)} NLTE model files to process")
        
        # Determine which files need processing
        files_to_process = []
        for file_path in nlte_files:
            filename = os.path.basename(file_path)
            if self.recalculate_all or filename not in existing_results:
                files_to_process.append(file_path)
        
        logger.info(f"Processing {len(files_to_process)} NLTE models (skipping {len(nlte_files) - len(files_to_process)} existing)")
        
        # Process files in parallel
        results = existing_results.copy()
        if files_to_process:
            try:
                with ThreadPoolExecutor(max_workers=self.threads) as executor:
                    future_to_file = {executor.submit(self._process_file, file_path): file_path for file_path in files_to_process}
                    for future in as_completed(future_to_file):
                        file_path = future_to_file[future]
                        filename = os.path.basename(file_path)
                        try:
                            model_result = future.result()
                            if model_result:
                                results[filename] = model_result
                                logger.debug(f"Completed quality calculation for {filename}")
                            else:
                                # Could be a special point or failed calculation
                                if any(sp['filename'] == filename for sp in self.special_points):
                                    logger.debug(f"Skipped special point: {filename}")
                                else:
                                    logger.warning(f"Failed to calculate quality for {filename}")
                        except Exception as e:
                            logger.error(f"Exception processing {filename}: {e}")
            except Exception as e:
                logger.error(f"Error during parallel processing: {e}")
                return None
        
        # Save normal results
        if results:
            try:
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4)
                logger.info(f"Saved quality results to {output_filename}")
                
                # Save special points to a separate file if any were found
                if self.special_points:
                    special_points_dir = os.path.join(os.path.dirname(self.quality_output_dir), "special_points")
                    os.makedirs(special_points_dir, exist_ok=True)
                    
                    special_points_file = os.path.join(
                        special_points_dir,
                        f"special_points_{timestamp}.json"
                    )
                    
                    special_points_data = {
                        "z_scale": self.special_points[0]['z_scale'] if self.special_points else "unknown",
                        "special_points": [
                            {
                                "teff": float(sp['teff']), 
                                "logg": float(sp['logg']), 
                                "quality": float(sp['Quality_shifted'])
                            } 
                            for sp in self.special_points
                        ],
                        "timestamp": timestamp,
                        "count": len(self.special_points)
                    }
                    
                    with open(special_points_file, 'w') as f:
                        json.dump(special_points_data, f, indent=4)
                    logger.info(f"Saved {len(self.special_points)} special points to {special_points_file}")
                
                return output_filename
            except Exception as e:
                logger.error(f"Failed to save quality results: {e}")
        
        return None
    
    def _process_file(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Process a single NLTE model file and calculate quality metrics."""
        filename = os.path.basename(file_path)
        
        try:
            # filename format: nlte{teff}-{logg}+{zscale in \d.\d format}.{model_version}.h5
            teff = filename.split('-')[0].replace('nlte', '')
            logg = filename.split('-')[1].split('+')[0]
            z_scale = filename.split('+')[1].split('.')[0] + '.' + filename.split('+')[1].split('.')[1].split('.')[0]
            
            # Check if Teff is divisible by 100, if not - save to special points and skip
            teff_float = float(teff)
            if teff_float % 100 != 0:
                logger.info(f"Skipping special point with Teff={teff} not divisible by 100: {filename}")
                # Save to special points list for later processing
                if not hasattr(self, 'special_points'):
                    self.special_points = []
                
                # Read H5 file to calculate quality anyway (for special points JSON)
                with h5py.File(file_path, 'r') as content:
                    wl = np.array(content['PHOENIX_SPECTRUM']['wl'][()])
                    # Convert from log10 flux to linear
                    flux = np.array(10.**content['PHOENIX_SPECTRUM']['flux'][()])
                
                quality_results = self._calculate_quality_value(flux, wl, self.org_flux, self.org_wl)
                if quality_results != 'Failed':
                    quality_shifted, quality_unshifted = quality_results
                    self.special_points.append({
                        'filename': filename,
                        'teff': teff,
                        'logg': logg,
                        'z_scale': z_scale,
                        'Quality_shifted': quality_shifted,
                        'Quality_unshifted': quality_unshifted
                    })
                return None  # Skip this point from regular results
            
            # Read H5 file for regular points
            with h5py.File(file_path, 'r') as content:
                wl = np.array(content['PHOENIX_SPECTRUM']['wl'][()])
                # Convert from log10 flux to linear
                flux = np.array(10.**content['PHOENIX_SPECTRUM']['flux'][()])
            
            # Calculate quality values
            quality_results = self._calculate_quality_value(flux, wl, self.org_flux, self.org_wl)
            
            if quality_results == 'Failed':
                logger.warning(f"Quality calculation failed for {filename}")
                return None
            
            quality_shifted, quality_unshifted = quality_results
            
            # Create result dictionary
            result = {
                'teff': teff,
                'logg': logg,
                'z_scale': z_scale,
                'Quality_shifted': quality_shifted,
                'Quality_unshifted': quality_unshifted,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return None
    
    def gauss_smooth(self, wl, fl, start, end, res, oversampling=50.0):
        """
        Smooth a data set (x,y) to a given resolution with a Gauss function.
        
        Args:
            wl: input wavelength (sorted, but not equally spaced)
            fl: fluxes (etc)
            start: beginning wavelength
            end: end wavelength
            res: target wavelength resolution (ca. FWHM)
            oversampling: (optional) oversampling factor, default 50.0
            
        Returns:
            wl_res, f_res: regular wavelength grid with F smoothed
        """
        # Make target wavelength array with oversampling
        n_points = int((end-start)*oversampling/res+1)
        wl_res = np.linspace((start), (end), n_points)
        
        # Interpolate input fluxes to wl_res (piecewise linear)
        fl_res = np.interp(wl_res, wl, fl)
        
        # Set up and perform FFT based convolution
        stddev = oversampling
        fl_smooth = convolve_fft(fl_res, Gaussian1DKernel(stddev))
        
        return wl_res, fl_smooth
    
    def normalize_flux(self, wavelengths, fluxes):
        """
        Normalize flux based on its integral.
        
        Args:
            wavelengths: Wavelength array
            fluxes: Flux array
            
        Returns:
            Normalized flux array
        """
        total_integral = np.trapezoid(fluxes, wavelengths)
        non_zero_indices = np.nonzero(fluxes)[0]
        
        if len(non_zero_indices) > 0:
            first = non_zero_indices[0]
            last = non_zero_indices[-1]
            distance = wavelengths[last] - wavelengths[first]
            normalized_fluxes = fluxes / total_integral * int(distance)
            return normalized_fluxes
            
        return fluxes  # Return original flux if no non-zero values
    
    def calculate_quality(self, original_flux, synthetic_flux):
        """
        Calculate quality metric between two spectra.
        
        Args:
            original_flux: Normalized flux of original spectrum
            synthetic_flux: Normalized flux of synthetic spectrum
            
        Returns:
            Quality value Q (mean squared error)
        """
        if len(original_flux) != len(synthetic_flux):
            raise ValueError("Both spectra must have the same length to calculate Q")
        
        # Calculate mean squared error
        Q = np.mean((original_flux - synthetic_flux) ** 2)
        return Q
    
    def interpolate_spectrum(self, wavelengths, fluxes, target_wavelengths):
        """
        Interpolate spectrum to target wavelengths.
        
        Args:
            wavelengths: Original wavelength array
            fluxes: Original flux array
            target_wavelengths: Target wavelength array
            
        Returns:
            Interpolated flux array
        """
        interp_func = interp1d(wavelengths, fluxes, kind="linear", bounds_error=False, fill_value=0)
        return interp_func(target_wavelengths)
    
    def _calculate_quality_value(self, flux, wl, org_flux, org_wl):
        """
        Calculate quality values between a synthetic and observed spectrum.
        
        Args:
            flux: Synthetic flux array
            wl: Synthetic wavelength array
            org_flux: Original (observed) flux array
            org_wl: Original wavelength array
            
        Returns:
            Tuple of (shifted_quality, unshifted_quality) or 'Failed' if calculation fails
        """
        try:
            # Variables needed
            res = 0.6
            xrange = [2000.0, 9000.0]
            
            # Apply Gaussian smoothing
            wl_smooth, flux_smooth = self.gauss_smooth(wl, flux, xrange[0]-res*2, xrange[1]+res*2, res)
            
            # Filter to wavelength range of interest (3800-4880 Å)
            filter_range = (3800, 4880)
            
            # Filter original data
            org_filtered_indices = (org_wl >= filter_range[0]) & (org_wl <= filter_range[1])
            org_wl_filtered = org_wl[org_filtered_indices]
            org_flux_filtered = org_flux[org_filtered_indices]
            
            # Filter synthetic data
            new_filtered_indices = (wl_smooth >= filter_range[0]) & (wl_smooth <= filter_range[1])
            wl_filtered = wl_smooth[new_filtered_indices]
            flux_filtered = flux_smooth[new_filtered_indices]
            
            # Define target wavelength grid (interpolation at 0.8 Å spacing)
            target_wavelengths = np.arange(3801, 4880, 0.8)
            
            # Calculate with wavelength shift
            # Wavelength shift factor (from original notebook)
            shift_factor = 0.999731
            shifted_wavelengths = wl_filtered * shift_factor
            
            # Interpolate original and shifted synthetic spectra to target wavelengths
            org_interpolated_flux = self.interpolate_spectrum(org_wl_filtered, org_flux_filtered, target_wavelengths)
            new_interpolated_flux_shifted = self.interpolate_spectrum(shifted_wavelengths, flux_filtered, target_wavelengths)
            
            # Normalize fluxes with shift
            org_normalized_flux_shifted = self.normalize_flux(target_wavelengths, org_interpolated_flux)
            new_normalized_flux_shifted = self.normalize_flux(target_wavelengths, new_interpolated_flux_shifted)
            
            # Calculate quality with shift
            Q_with_shift = self.calculate_quality(org_normalized_flux_shifted, new_normalized_flux_shifted)
            
            # Calculate without wavelength shift
            org_interpolated_flux_no_shift = self.interpolate_spectrum(org_wl_filtered, org_flux_filtered, target_wavelengths)
            new_interpolated_flux_no_shift = self.interpolate_spectrum(wl_filtered, flux_filtered, target_wavelengths)
            
            # Normalize fluxes without shift
            org_normalized_flux_no_shift = self.normalize_flux(target_wavelengths, org_interpolated_flux_no_shift)
            new_normalized_flux_no_shift = self.normalize_flux(target_wavelengths, new_interpolated_flux_no_shift)
            
            # Calculate quality without shift
            Q_without_shift = self.calculate_quality(org_normalized_flux_no_shift, new_normalized_flux_no_shift)
            
            return (Q_with_shift, Q_without_shift)
            
        except ValueError as e:
            logger.error(f"Error calculating quality: {e}")
            return 'Failed'
        except Exception as e:
            logger.error(f"Unexpected error in quality calculation: {e}")
            return 'Failed'