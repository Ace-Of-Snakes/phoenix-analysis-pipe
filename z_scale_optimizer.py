#!/usr/bin/env python3
# z_scale_optimizer.py

import os
import json
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.optimize import minimize, minimize_scalar, curve_fit
from scipy.interpolate import interp1d, UnivariateSpline
from astropy.convolution import convolve_fft, Gaussian1DKernel
from tqdm import tqdm

logger = logging.getLogger("ZScaleOptimizer")

class ZScaleOptimizer:
    """
    Analyze models with different z-scale values to find optimal z-scale.
    """
    
    def __init__(self, z_scale_models_dir: str, quality_output_dir: str, 
                 image_output_dir: str, report_output_dir: str, threads: int = 4):
        """
        Initialize the Z-scale optimizer.
        
        Args:
            z_scale_models_dir: Directory containing models with different z-scales
            quality_output_dir: Directory for storing quality calculation results
            image_output_dir: Directory for storing output images
            report_output_dir: Directory for storing reports
            threads: Number of threads to use for parallel processing
        """
        self.z_scale_models_dir = z_scale_models_dir
        self.quality_output_dir = quality_output_dir
        self.image_output_dir = image_output_dir
        self.report_output_dir = report_output_dir
        self.threads = threads
        
        # Ensure output directories exist
        os.makedirs(self.quality_output_dir, exist_ok=True)
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.report_output_dir, "z_scale"), exist_ok=True)
        
        # Load needed modules from quality_calculator for spectral calculations
        from quality_calculator import QualityCalculator
        self.quality_calculator = QualityCalculator(
            nlte_models_dir=z_scale_models_dir,
            quality_output_dir=quality_output_dir,
            original_spectrum_file="uves_spectra_fomalhaut.csv"
        )
    
    def run(self) -> Optional[str]:
        """
        Run z-scale optimization analysis.
        
        Returns:
            Path to the results JSON file, or None if failed
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Get list of H5 files in the z-scale models directory
            h5_files = [f for f in os.listdir(self.z_scale_models_dir) 
                      if f.endswith('.h5') and 'nlte' in f]
            
            if not h5_files:
                logger.error(f"No H5 files found in {self.z_scale_models_dir}")
                return None
                
            logger.info(f"Found {len(h5_files)} H5 files for z-scale analysis")
            
            # Extract z-scale information and group by teff/logg
            teff_logg_pairs = {}
            for file in h5_files:
                try:
                    # filename format: nlte{teff}-{logg}+{zscale in \d.\d format}.{model_version}.h5
                    teff = file.split('-')[0].replace('nlte', '')
                    logg = file.split('-')[1].split('+')[0]
                    z_scale = file.split('+')[1].split('.')[0] + '.' + file.split('+')[1].split('.')[1].split('.')[0]
                    
                    key = (teff, logg)
                    if key not in teff_logg_pairs:
                        teff_logg_pairs[key] = []
                    
                    teff_logg_pairs[key].append((z_scale, os.path.join(self.z_scale_models_dir, file)))
                
                except (IndexError, ValueError) as e:
                    logger.warning(f"Skipping file with invalid format: {file} - {e}")
            
            logger.info(f"Found {len(teff_logg_pairs)} teff/logg pairs with varying z-scales")
            
            # Filter to pairs with enough z-scale values for analysis
            valid_pairs = {k: v for k, v in teff_logg_pairs.items() if len(v) >= 4}
            
            if not valid_pairs:
                logger.error("No teff/logg pairs have enough z-scale values for analysis (need at least 4)")
                return None
            
            logger.info(f"Found {len(valid_pairs)} teff/logg pairs with sufficient z-scale values")
            
            # Calculate quality values for each model
            quality_results = {}
            with ThreadPoolExecutor(max_workers=self.threads) as executor:
                # Create a list of all files to process
                files_to_process = []
                for teff_logg, z_scale_files in valid_pairs.items():
                    for z_scale, file_path in z_scale_files:
                        files_to_process.append((teff_logg, z_scale, file_path))
                
                # Process files in parallel
                future_to_file = {executor.submit(self._process_file, teff_logg, z_scale, file_path): 
                                 (teff_logg, z_scale, file_path) for teff_logg, z_scale, file_path in files_to_process}
                
                # Collect results as they complete
                for future in as_completed(future_to_file):
                    teff_logg, z_scale, file_path = future_to_file[future]
                    try:
                        result = future.result()
                        if result:
                            if teff_logg not in quality_results:
                                quality_results[teff_logg] = []
                            quality_results[teff_logg].append((z_scale, result))
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {e}")
            
            if not quality_results:
                logger.error("Failed to calculate quality values for z-scale analysis")
                return None
            
            # Find optimal z-scale for each teff/logg pair
            z_scale_optima = {}
            for teff_logg, results in quality_results.items():
                # Sort by z_scale as float
                results.sort(key=lambda x: float(x[0]))
                
                z_scales = np.array([float(z[0]) for z in results])
                qualities = np.array([float(q[1]['Quality_shifted']) for q in results])
                
                # Find minimum using ensemble of methods
                min_results = self._find_optimal_z_scale_ensemble(z_scales, qualities, teff_logg)
                if min_results:
                    z_scale_optima[teff_logg] = min_results
            
            # Create output directory for results
            output_dir = os.path.join(self.report_output_dir, "z_scale")
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate visualization
            output_image = os.path.join(
                self.image_output_dir, 
                f"z_scale_analysis_{timestamp}.png"
            )
            
            self._visualize_results(z_scale_optima, output_image)
            
            # Save results to JSON
            json_output = os.path.join(
                output_dir,
                f"z_scale_analysis_{timestamp}.json"
            )
            
            with open(json_output, 'w', encoding='utf-8') as f:
                # Convert any remaining NumPy values to Python types
                serializable_results = {}
                for k, v in z_scale_optima.items():
                    key = str(k)  # Convert tuple key to string
                    serializable_results[key] = self._convert_to_serializable(v)
                json.dump(serializable_results, f, indent=4)
            
            # Generate report
            report_file = os.path.join(
                output_dir,
                f"z_scale_analysis_{timestamp}.txt"
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"Z-SCALE OPTIMIZATION ANALYSIS RESULTS\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"OVERVIEW:\n")
                f.write(f"- Analyzed {len(quality_results)} teff/logg pairs\n")
                f.write(f"- Found optimal z-scale for {len(z_scale_optima)} pairs\n\n")
                
                f.write(f"DETAILED RESULTS:\n")
                f.write(f"=================\n\n")
                
                for teff_logg, result in z_scale_optima.items():
                    teff, logg = teff_logg
                    f.write(f"Teff={teff}, logg={logg}:\n")
                    f.write(f"  Optimal z-scale: {result['min_z_scale']:.4f} ± {result.get('min_z_scale_std', 0.0):.4f}\n")
                    f.write(f"  Minimum quality: {result['min_quality']:.8f}\n")
                    
                    # Show all methods and their results
                    f.write(f"  Method results:\n")
                    method_results = result.get('method_results', [])
                    for method_name, method_z_scale, method_quality in method_results:
                        f.write(f"    {method_name}: z-scale={method_z_scale:.4f}, quality={method_quality:.8f}\n")
                    
                    f.write(f"  Analyzed z-scales: {', '.join([f'{z:.4f}' for z in result['z_scales']])}\n\n")
                
                # Summary of recommendations
                f.write(f"RECOMMENDATIONS:\n")
                f.write(f"===============\n\n")
                
                z_scale_values = [result['min_z_scale'] for result in z_scale_optima.values()]
                z_scale_weights = [1.0/result.get('min_z_scale_std', 0.0001) for result in z_scale_optima.values()]
                
                # Calculate weighted mean and standard deviation
                weighted_mean = np.average(z_scale_values, weights=z_scale_weights)
                weighted_std = np.sqrt(np.average((np.array(z_scale_values) - weighted_mean)**2, weights=z_scale_weights))
                
                f.write(f"Average optimal z-scale across all models (weighted): {weighted_mean:.4f} ± {weighted_std:.4f}\n\n")
                
                if weighted_std < 0.1:
                    f.write("The optimal z-scale is fairly consistent across models, suggesting a global optimum.\n")
                    f.write(f"Recommended z-scale for future models: {weighted_mean:.4f}\n")
                else:
                    f.write("The optimal z-scale varies significantly across models.\n")
                    f.write("Consider using different z-scales for different temperature/gravity regimes.\n")
                    
                    # Group by Teff
                    teff_groups = {}
                    for (teff, logg), result in z_scale_optima.items():
                        if teff not in teff_groups:
                            teff_groups[teff] = []
                        teff_groups[teff].append((result['min_z_scale'], result.get('min_z_scale_std', 0.0001)))
                    
                    f.write("\nZ-scale recommendations by temperature:\n")
                    for teff, scales_and_stds in teff_groups.items():
                        scales = [scale for scale, _ in scales_and_stds]
                        weights = [1.0/std for _, std in scales_and_stds]
                        mean_teff_scale = np.average(scales, weights=weights)
                        f.write(f"  Teff={teff}: {mean_teff_scale:.4f}\n")
            
            logger.info(f"Z-scale analysis report saved to {report_file}")
            
            return json_output
            
        except Exception as e:
            logger.error(f"Error in z-scale optimization: {e}")
            return None
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(i) for i in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_serializable(i) for i in obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        else:
            return obj
    
    def _process_file(self, teff_logg: Tuple[str, str], z_scale: str, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single H5 file and calculate quality metrics.
        
        Args:
            teff_logg: Tuple of (teff, logg) strings
            z_scale: Z-scale value as string
            file_path: Path to the H5 file
            
        Returns:
            Dictionary with quality results or None if failed
        """
        filename = os.path.basename(file_path)
        
        try:
            # Read H5 file
            with h5py.File(file_path, 'r') as content:
                wl = np.array(content['PHOENIX_SPECTRUM']['wl'][()])
                # Convert from log10 flux to linear
                flux = np.array(10.**content['PHOENIX_SPECTRUM']['flux'][()])
            
            # Calculate quality values using the quality calculator
            quality_results = self.quality_calculator._calculate_quality_value(
                flux, wl, self.quality_calculator.org_flux, self.quality_calculator.org_wl)
            
            if quality_results == 'Failed':
                logger.warning(f"Quality calculation failed for {filename}")
                return None
            
            quality_shifted, quality_unshifted = quality_results
            
            # Create result dictionary
            result = {
                'teff': teff_logg[0],
                'logg': teff_logg[1],
                'z_scale': z_scale,
                'Quality_shifted': quality_shifted,
                'Quality_unshifted': quality_unshifted,
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            return None
    
    def _find_min_with_focused_quadratic(self, z_scales: np.ndarray, qualities: np.ndarray) -> Tuple[float, float, str]:
        """
        Find minimum using a quadratic fit focused on the region near the minimum.
        
        Args:
            z_scales: Array of z-scale values
            qualities: Array of quality values
            
        Returns:
            min_z_scale: Estimated optimal z-scale
            min_quality: Estimated minimum quality value
            method_name: Name of the method used
        """
        try:
            # Get region around the apparent minimum
            min_idx = np.argmin(qualities)
            min_z = z_scales[min_idx]
            
            # Focus on data points around the minimum (within ±50% of the full range)
            z_range = np.max(z_scales) - np.min(z_scales)
            focus_radius = 0.5 * z_range
            
            min_region_mask = (z_scales >= min_z - focus_radius) & (z_scales <= min_z + focus_radius)
            
            # Ensure we have at least 3 points for quadratic fit
            if np.sum(min_region_mask) >= 3:
                z_focused = z_scales[min_region_mask]
                q_focused = qualities[min_region_mask]
                
                # Fit quadratic: a*x^2 + b*x + c
                coefs = np.polyfit(z_focused, q_focused, 2)
                a, b, c = coefs
                
                # If a > 0, we have a proper minimum 
                if a > 0:
                    # Minimum of quadratic is at x = -b/(2a)
                    min_z_scale = -b / (2 * a)
                    
                    # Check if the minimum is within the data range
                    if min_z_scale < np.min(z_scales) or min_z_scale > np.max(z_scales):
                        # If outside range, fallback to the data minimum
                        return z_scales[min_idx], qualities[min_idx], "Focused Quadratic (out of range)"
                    
                    # Calculate the quality at the minimum z_scale
                    min_quality = a * min_z_scale**2 + b * min_z_scale + c
                    return min_z_scale, min_quality, "Focused Quadratic"
            
            # If we don't have enough points or a doesn't indicate a minimum
            return z_scales[min_idx], qualities[min_idx], "Data Minimum (quadratic fallback)"
            
        except Exception as e:
            logger.warning(f"Error in focused quadratic fit: {e}")
            min_idx = np.argmin(qualities)
            return z_scales[min_idx], qualities[min_idx], "Data Minimum (quadratic error)"
    
    def _find_min_with_spline(self, z_scales: np.ndarray, qualities: np.ndarray) -> Tuple[float, float, str]:
        """
        Find minimum using spline interpolation.
        
        Args:
            z_scales: Array of z-scale values
            qualities: Array of quality values
            
        Returns:
            min_z_scale: Estimated optimal z-scale
            min_quality: Estimated minimum quality value
            method_name: Name of the method used
        """
        try:
            # Need at least 4 points for cubic spline
            if len(z_scales) < 4:
                min_idx = np.argmin(qualities)
                return z_scales[min_idx], qualities[min_idx], "Data Minimum (not enough points for spline)"
            
            # Sort the data
            sort_idx = np.argsort(z_scales)
            z_sorted = z_scales[sort_idx]
            q_sorted = qualities[sort_idx]
            
            # Create a spline with appropriate smoothing
            # Use a small smoothing factor to prevent overfitting
            s = 0.0001 * len(z_scales)  # Adaptive smoothing based on number of points
            spline = UnivariateSpline(z_sorted, q_sorted, k=3, s=s)
            
            # Create a dense grid to find the minimum precisely
            z_dense = np.linspace(np.min(z_scales), np.max(z_scales), 1000)
            q_dense = spline(z_dense)
            
            # Find the minimum
            min_idx = np.argmin(q_dense)
            min_z_scale = z_dense[min_idx]
            min_quality = q_dense[min_idx]
            
            return min_z_scale, min_quality, "Spline Interpolation"
            
        except Exception as e:
            logger.warning(f"Error in spline interpolation: {e}")
            min_idx = np.argmin(qualities)
            return z_scales[min_idx], qualities[min_idx], "Data Minimum (spline error)"
    
    def _find_min_with_direct_optimization(self, z_scales: np.ndarray, qualities: np.ndarray) -> Tuple[float, float, str]:
        """
        Find minimum using direct optimization.
        
        Args:
            z_scales: Array of z-scale values
            qualities: Array of quality values
            
        Returns:
            min_z_scale: Estimated optimal z-scale
            min_quality: Estimated minimum quality value
            method_name: Name of the method used
        """
        try:
            # Need enough points for a reliable interpolation
            if len(z_scales) < 4:
                min_idx = np.argmin(qualities)
                return z_scales[min_idx], qualities[min_idx], "Data Minimum (not enough points)"
            
            # Sort values for interpolation
            sort_idx = np.argsort(z_scales)
            z_sorted = z_scales[sort_idx]
            q_sorted = qualities[sort_idx]
            
            # Create an interpolation function
            interp_func = interp1d(z_sorted, q_sorted, kind='cubic', bounds_error=False, 
                                fill_value=(q_sorted[0], q_sorted[-1]))
            
            # Function to minimize
            def objective(x):
                return float(interp_func(x))
            
            # Use Brent's method for bounded optimization
            result = minimize_scalar(objective, 
                                  bounds=(np.min(z_scales), np.max(z_scales)),
                                  method='bounded')
            
            if result.success:
                min_z_scale = result.x
                min_quality = result.fun
                return min_z_scale, min_quality, "Direct Optimization"
            else:
                # Fallback to data minimum if optimization fails
                min_idx = np.argmin(qualities)
                return z_scales[min_idx], qualities[min_idx], "Data Minimum (optimization failed)"
                
        except Exception as e:
            logger.warning(f"Error in direct optimization: {e}")
            min_idx = np.argmin(qualities)
            return z_scales[min_idx], qualities[min_idx], "Data Minimum (optimization error)"
    
    def _find_optimal_z_scale_ensemble(self, z_scales: np.ndarray, qualities: np.ndarray, 
                                     teff_logg: Tuple[str, str]) -> Dict[str, Any]:
        """
        Find optimal z-scale using an ensemble of methods for improved reliability.
        
        Args:
            z_scales: Array of z-scale values
            qualities: Array of quality values
            teff_logg: Tuple of (teff, logg) strings for logging
            
        Returns:
            Dictionary with optimization results and uncertainty estimates
        """
        logger.info(f"Finding optimal z-scale for Teff={teff_logg[0]}, logg={teff_logg[1]}")
        
        # Collect results from multiple methods
        method_results = []
        
        # Method 1: Find minimum directly from data
        min_idx = np.argmin(qualities)
        min_z_data = z_scales[min_idx]
        min_q_data = qualities[min_idx]
        method_results.append(("Data Minimum", min_z_data, min_q_data))
        
        # Method 2: Focused quadratic fit
        min_z_quad, min_q_quad, quad_method = self._find_min_with_focused_quadratic(z_scales, qualities)
        method_results.append((quad_method, min_z_quad, min_q_quad))
        
        # Method 3: Spline interpolation
        min_z_spline, min_q_spline, spline_method = self._find_min_with_spline(z_scales, qualities)
        method_results.append((spline_method, min_z_spline, min_q_spline))
        
        # Method 4: Direct optimization
        min_z_opt, min_q_opt, opt_method = self._find_min_with_direct_optimization(z_scales, qualities)
        method_results.append((opt_method, min_z_opt, min_q_opt))
        
        # Calculate statistics from valid results
        valid_z_values = [result[1] for result in method_results 
                        if np.min(z_scales) <= result[1] <= np.max(z_scales)]
        
        if not valid_z_values:
            # If no valid results, use the data minimum
            logger.warning(f"No valid z-scale minimum found, using data minimum")
            min_z_scale = min_z_data
            min_quality = min_q_data
            min_z_std = 0.0
        else:
            # Calculate weighted mean based on quality values
            # Lower quality values get higher weights
            z_values = np.array(valid_z_values)
            q_values = np.array([result[2] for idx, result in enumerate(method_results) 
                              if np.min(z_scales) <= method_results[idx][1] <= np.max(z_scales)])
            
            # Invert qualities for weights (lower quality → higher weight)
            # and normalize
            quality_range = np.max(qualities) - np.min(qualities)
            if quality_range > 0:
                weights = (np.max(qualities) - q_values) / quality_range
                weights = weights / np.sum(weights)
                
                # Calculate weighted mean and standard deviation
                min_z_scale = np.sum(z_values * weights)
                min_z_std = np.sqrt(np.sum(weights * ((z_values - min_z_scale)**2)))
                
                # Estimate the quality at the final z_scale using the best method
                best_method_idx = np.argmin([result[2] for result in method_results])
                best_method = method_results[best_method_idx][0]
                
                if best_method.startswith("Spline"):
                    # Use spline to predict quality
                    sort_idx = np.argsort(z_scales)
                    spline = UnivariateSpline(z_scales[sort_idx], qualities[sort_idx], k=3, s=0.0001)
                    min_quality = float(spline(min_z_scale))
                elif best_method.startswith("Focused Quadratic"):
                    # Use quadratic to predict quality
                    min_region_mask = (z_scales >= np.min(z_scales)) & (z_scales <= np.max(z_scales))
                    z_focused = z_scales[min_region_mask]
                    q_focused = qualities[min_region_mask]
                    coefs = np.polyfit(z_focused, q_focused, 2)
                    a, b, c = coefs
                    min_quality = a * min_z_scale**2 + b * min_z_scale + c
                else:
                    # Use linear interpolation
                    sort_idx = np.argsort(z_scales)
                    z_sorted = z_scales[sort_idx]
                    q_sorted = qualities[sort_idx]
                    interp = interp1d(z_sorted, q_sorted, kind='linear', bounds_error=False, 
                                     fill_value=(q_sorted[0], q_sorted[-1]))
                    min_quality = float(interp(min_z_scale))
            else:
                # If all qualities are the same, use the mean z-scale
                min_z_scale = np.mean(z_values)
                min_z_std = np.std(z_values)
                min_quality = np.mean(q_values)
        
        # If standard deviation is zero, set a small default value
        if min_z_std == 0:
            min_z_std = 0.01 * min_z_scale
        
        # Create result dictionary
        result = {
            'z_scales': z_scales.tolist(),
            'qualities': qualities.tolist(),
            'min_z_scale': float(min_z_scale),
            'min_z_scale_std': float(min_z_std),
            'min_quality': float(min_quality),
            'method_results': [(name, float(z), float(q)) for name, z, q in method_results]
        }
        
        logger.info(f"Optimal z-scale for Teff={teff_logg[0]}, logg={teff_logg[1]}: {min_z_scale:.4f} ± {min_z_std:.4f}")
        return result
    
    def _visualize_results(self, z_scale_optima: Dict[Tuple[str, str], Dict[str, Any]], output_file: str):
        """
        Create a visualization of z-scale optimization results.
        
        Args:
            z_scale_optima: Dictionary with optimization results
            output_file: Path to save the output image
        """
        # Count the number of teff/logg pairs
        n_pairs = len(z_scale_optima)
        
        # Create figure with subplots
        n_cols = min(3, n_pairs)
        n_rows = (n_pairs + n_cols - 1) // n_cols  # Ceiling division
        figsize = (5 * n_cols, 4 * n_rows + 1)  # Extra space for title
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # If there's only one subplot, axes isn't an array
        if n_pairs == 1:
            axes = np.array([[axes]])
        # If there's only one row, axes is 1D
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Get a summary of all z_scale values for the title
        all_min_z_scales = [result['min_z_scale'] for result in z_scale_optima.values()]
        all_min_z_stds = [result.get('min_z_scale_std', 0.0001) for result in z_scale_optima.values()]
        weights = [1.0/std for std in all_min_z_stds]
        
        weighted_mean = np.average(all_min_z_scales, weights=weights)
        weighted_std = np.sqrt(np.average((np.array(all_min_z_scales) - weighted_mean)**2, weights=weights))
        
        # Process each teff/logg pair
        for i, ((teff, logg), result) in enumerate(z_scale_optima.items()):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            # Extract data
            z_scales = np.array(result['z_scales'])
            qualities = np.array(result['qualities'])
            min_z_scale = result['min_z_scale']
            min_quality = result['min_quality']
            min_z_std = result.get('min_z_scale_std', 0.0001)
            
            # Sort z_scales and qualities together
            sort_idx = np.argsort(z_scales)
            z_scales_sorted = z_scales[sort_idx]
            qualities_sorted = qualities[sort_idx]
            
            # Create smooth interpolation for the line
            z_range = np.max(z_scales) - np.min(z_scales)
            z_grid = np.linspace(np.min(z_scales) - 0.1*z_range, np.max(z_scales) + 0.1*z_range, 1000)
            
            # Choose interpolation method based on number of points
            try:
                if len(z_scales) >= 4:
                    # Try spline interpolation
                    spline = UnivariateSpline(z_scales_sorted, qualities_sorted, k=3, s=0.0001)
                    quality_grid = spline(z_grid)
                    
                    # Plot the smooth line
                    ax.plot(z_grid, quality_grid, '-', color='blue', lw=2, alpha=0.7)
                else:
                    # Fallback to simple linear interpolation
                    interp = interp1d(z_scales_sorted, qualities_sorted, 
                                    kind='linear', bounds_error=False, fill_value='extrapolate')
                    quality_grid = interp(z_grid)
                    ax.plot(z_grid, quality_grid, '-', color='blue', lw=2, alpha=0.7)
            except:
                # With few points, just connect the dots
                ax.plot(z_scales_sorted, qualities_sorted, '-', color='blue', lw=2, alpha=0.7)
            
            # Plot original data points
            ax.scatter(z_scales, qualities, color='blue', s=60, zorder=10, label='Measured quality')
            
            # Highlight the minimum with uncertainty
            ax.scatter([min_z_scale], [min_quality], color='red', s=100, zorder=11, 
                     edgecolor='black', label=f'Minimum: {min_z_scale:.4f} ± {min_z_std:.4f}')
            
            # Draw vertical line at minimum
            ax.axvline(min_z_scale, color='red', linestyle='--', alpha=0.5)
            
            # Draw vertical span for uncertainty
            ax.axvspan(min_z_scale - min_z_std, min_z_scale + min_z_std, alpha=0.2, color='red')
            
            # Plot individual method results if available
            method_results = result.get('method_results', [])
            if method_results:
                # For each method, plot a small marker
                for method_name, method_z, method_q in method_results:
                    if method_name == "Data Minimum":
                        marker = 'X'
                        color = 'darkblue'
                    elif method_name.startswith("Focused Quadratic"):
                        marker = 's'
                        color = 'green'
                    elif method_name.startswith("Spline"):
                        marker = '^'
                        color = 'purple'
                    elif method_name.startswith("Direct"):
                        marker = 'D'
                        color = 'orange'
                    else:
                        marker = 'o'
                        color = 'gray'
                    
                    ax.scatter([method_z], [method_q], marker=marker, color=color, s=40, alpha=0.7,
                            edgecolor='black', linewidth=0.5)
            
            # Set labels and title
            ax.set_xlabel('Z-Scale', fontsize=12)
            ax.set_ylabel('Quality (Shifted)', fontsize=12)
            ax.set_title(f'Teff={teff}, logg={logg}', fontsize=14)
            
            # Add grid and legend
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10, loc='best')
        
        # Hide unused subplots
        for i in range(n_pairs, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].axis('off')
        
        # Overall title
        plt.suptitle(f'Z-Scale Optimization Analysis\nWeighted mean optimal z-scale: {weighted_mean:.4f} ± {weighted_std:.4f}', 
                   fontsize=16)
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for suptitle
        
        # Save figure
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Z-scale visualization saved to {output_file}")