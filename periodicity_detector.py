#!/usr/bin/env python3
# enhanced_periodicity_detector.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import signal
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN, KMeans
from scipy.fft import fft, fftfreq
from scipy import ndimage
from scipy.spatial import Delaunay
from scipy.stats import binned_statistic_2d

logger = logging.getLogger("PeriodicityDetector")

class PeriodicityDetector:
    """
    Enhanced analysis for detecting and visualizing periodicities in heatmap data,
    with improved pattern detection and visualization capabilities.
    """
    
    def __init__(self, quality_output_dir: str, image_output_dir: str, report_output_dir: str):
        """
        Initialize the enhanced periodicity detector.
        
        Args:
            quality_output_dir: Directory containing quality calculation results
            image_output_dir: Directory for storing output images
            report_output_dir: Directory for storing reports
        """
        self.quality_output_dir = quality_output_dir
        self.image_output_dir = image_output_dir
        self.report_output_dir = report_output_dir
        
        # Ensure output directories exist
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.report_output_dir, "periodicities"), exist_ok=True)
    
    def run(self, quality_results_file: str) -> Optional[str]:
        """
        Run enhanced periodicity detection with improved pattern visualization.
        
        Args:
            quality_results_file: Path to JSON file with quality calculation results
            
        Returns:
            Path to the results report file, or None if failed
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Load data
            data = self.load_data(quality_results_file)
            if not data:
                logger.error("Failed to load quality data")
                return None
            
            # Extract grid data
            teff_values, logg_values, quality_shifted_values, z_scale_values = self.extract_grid_data(data)
            
            if len(teff_values) < 5:
                logger.error("Not enough data points for periodicity analysis")
                return None
            
            # Get z_scale for reporting
            z_scale = z_scale_values[0] if z_scale_values.any() else "N/A"

            # Detect periodicities with enhanced techniques
            x_periodicity, y_periodicity, x_std, y_std = self.enhanced_periodicity_detection(
                teff_values, logg_values, quality_shifted_values)
            
            # Find patterns in quality values with improved clustering
            cluster_labels, n_clusters = self.enhanced_pattern_detection(
                teff_values, logg_values, quality_shifted_values)
            
            # Extract data patterns and features
            pattern_features = self.extract_pattern_features(
                teff_values, logg_values, quality_shifted_values)
            
            # Get ranges for prediction
            x_range = (min(teff_values), max(teff_values))
            y_range = (min(logg_values), max(logg_values))
            
            # Predict additional points using intelligent pattern-based approach
            pred_x, pred_y = self.pattern_based_prediction(
                teff_values, logg_values, quality_shifted_values, 
                x_periodicity, y_periodicity, pattern_features,
                x_range, y_range, n_points=200)
            
            # Create enhanced visualization with pattern highlighting
            output_image = os.path.join(
                self.image_output_dir, 
                f"periodicities_{timestamp}.png"
            )
            
            self.enhanced_visualization(
                teff_values, logg_values, quality_shifted_values,
                pred_x, pred_y, x_periodicity, y_periodicity,
                cluster_labels, n_clusters, pattern_features, 
                z_scale, output_image,
                x_range=x_range, y_range=y_range
            )
            
            # Export predicted points to CSV
            csv_filename = os.path.join(
                self.report_output_dir,
                "periodicities",
                f"predicted_points_{timestamp}.csv"
            )
            
            if len(pred_x) > 0:
                with open(csv_filename, 'w') as f:
                    f.write("Teff,logg\n")
                    for i in range(len(pred_x)):
                        f.write(f"{pred_x[i]:.2f},{pred_y[i]:.4f}\n")
                logger.info(f"Predicted points exported to {csv_filename}")
            
            # Create enhanced report with pattern insights
            report_file = os.path.join(
                self.report_output_dir,
                "periodicities",
                f"periodicity_analysis_{timestamp}.txt"
            )
            
            with open(report_file, 'w') as f:
                f.write("=== ENHANCED PERIODICITY ANALYSIS RESULTS ===\n")
                f.write(f"Input file: {quality_results_file}\n")
                f.write(f"Data points: {len(teff_values)}\n")
                f.write(f"Z-scale: {z_scale}\n\n")
                
                f.write("--- Detected Periodicities ---\n")
                f.write(f"Teff periodicity: {x_periodicity:.2f} ± {x_std:.2f}\n")
                f.write(f"logg periodicity: {y_periodicity:.4f} ± {y_std:.4f}\n")
                f.write(f"Number of pattern clusters: {n_clusters}\n")
                f.write(f"Suggested additional data points: {len(pred_x)}\n\n")
                
                # Pattern insights section
                f.write("--- Pattern Insights ---\n")
                if pattern_features.get('quality_gradient_regions'):
                    f.write("Detected Quality Gradient Regions:\n")
                    for i, region in enumerate(pattern_features['quality_gradient_regions'][:5]):
                        f.write(f"  Region {i+1}: Teff = {region[0]:.1f}-{region[1]:.1f}, logg = {region[2]:.2f}-{region[3]:.2f}\n")
                
                if pattern_features.get('teff_sensitivity') and pattern_features.get('logg_sensitivity'):
                    f.write(f"\nQuality Sensitivity Analysis:\n")
                    f.write(f"  Teff sensitivity: {pattern_features['teff_sensitivity']:.6f} (quality change per 100K)\n")
                    f.write(f"  logg sensitivity: {pattern_features['logg_sensitivity']:.6f} (quality change per 0.1 dex)\n")
                    
                    if abs(pattern_features['teff_sensitivity']) > abs(pattern_features['logg_sensitivity']):
                        f.write("  Quality is more sensitive to changes in Teff than logg\n")
                    else:
                        f.write("  Quality is more sensitive to changes in logg than Teff\n")
                
                if 'symmetry_score' in pattern_features:
                    f.write(f"\nPattern Symmetry Analysis:\n")
                    f.write(f"  Symmetry score: {pattern_features['symmetry_score']:.2f} (0-1 scale)\n")
                    if pattern_features['symmetry_score'] > 0.7:
                        f.write("  High symmetry detected - suggests regular physical patterns\n")
                    elif pattern_features['symmetry_score'] > 0.4:
                        f.write("  Moderate symmetry detected - some regular patterns present\n")
                    else:
                        f.write("  Low symmetry detected - complex or irregular patterns\n")
                        
                # Group predictions by Teff
                if len(pred_x) > 0:
                    teff_groups = {}
                    for i in range(len(pred_x)):
                        teff = pred_x[i]
                        logg = pred_y[i]
                        if teff not in teff_groups:
                            teff_groups[teff] = []
                        teff_groups[teff].append(logg)
                    
                    f.write("\nMost promising regions to explore:\n")
                    for i, (teff, loggs) in enumerate(sorted(teff_groups.items())[:5]):
                        logg_min, logg_max = min(loggs), max(loggs)
                        f.write(f"  Region {i+1}: Teff = {teff:.1f}, logg range: {logg_min:.2f}-{logg_max:.2f}\n")
                
                # Add enhanced recommendations
                f.write("\n=== ENHANCED RECOMMENDATIONS ===\n\n")
                f.write("1. PATTERN-BASED SAMPLING APPROACH:\n")
                
                if n_clusters > 0:
                    f.write("   The data shows distinct cluster patterns. We recommend focused sampling in these clustered regions.\n")
                    
                    # Cluster-specific recommendations
                    for i in range(n_clusters):
                        mask = cluster_labels == i
                        if np.any(mask):
                            cluster_teff = teff_values[mask]
                            cluster_logg = logg_values[mask]
                            cluster_quality = quality_shifted_values[mask]
                            
                            # Find highest quality point in cluster
                            best_idx = np.argmin(cluster_quality)
                            
                            f.write(f"   Cluster {i+1}: Center at Teff={np.mean(cluster_teff):.1f}, logg={np.mean(cluster_logg):.3f}\n")
                            f.write(f"     - Best quality: {cluster_quality[best_idx]:.6f} at Teff={cluster_teff[best_idx]:.1f}, logg={cluster_logg[best_idx]:.3f}\n")
                            f.write(f"     - Recommended sampling: {max(3, len(cluster_teff)//2)} points around this region\n")
                else:
                    f.write("   No strong clustering detected. We recommend sampling according to the detected periodicities:\n")
                    f.write(f"     - Sample at Teff intervals of approximately {x_periodicity:.1f}\n")
                    f.write(f"     - Sample at logg intervals of approximately {y_periodicity:.4f}\n")
                
                f.write("\n2. GRADIENT EXPLORATION STRATEGY:\n")
                if pattern_features.get('quality_gradient_regions'):
                    f.write("   Focus on regions with high quality gradients for maximum information gain:\n")
                    for i, region in enumerate(pattern_features['quality_gradient_regions'][:3]):
                        f.write(f"     - Region {i+1}: Teff = {region[0]:.1f}-{region[1]:.1f}, logg = {region[2]:.2f}-{region[3]:.2f}\n")
                        f.write(f"       Recommended: 5-7 points along the steepest gradient direction\n")
                else:
                    f.write("   No significant quality gradients detected. Use regular grid sampling.\n")
                
                f.write("\n3. PERIODICTY VERIFICATION STRATEGY:\n")
                f.write(f"   To verify the detected periodicities (Teff={x_periodicity:.1f}, logg={y_periodicity:.4f}):\n")
                f.write(f"     - Sample points at 1/2, 1, 3/2, and 2 multiples of the periodicity\n")
                f.write(f"     - Key Teff values: {x_periodicity/2:.1f}, {x_periodicity:.1f}, {x_periodicity*1.5:.1f}, {x_periodicity*2:.1f}\n")
                f.write(f"     - Key logg values: {y_periodicity/2:.4f}, {y_periodicity:.4f}, {y_periodicity*1.5:.4f}, {y_periodicity*2:.4f}\n")
                
                # Final recommendations
                f.write("\n4. OPTIMAL SAMPLING PLAN:\n")
                f.write(f"   For best results, collect at least {max(10, len(pred_x) // 10)} additional data points.\n")
                f.write(f"   Use the predicted points in the exported CSV file as a starting guide.\n")
                f.write(f"   Allocate samples in this ratio:\n")
                f.write(f"     - 40%: High gradient regions for maximum information gain\n")
                f.write(f"     - 30%: Cluster verification and refinement\n")
                f.write(f"     - 30%: Periodicity verification at key intervals\n")
            
            logger.info(f"Enhanced periodicity analysis report saved to {report_file}")
            
            # Also save results as JSON for use by other components
            json_results = {
                "z_scale": z_scale,
                "teff_periodicity": x_periodicity,
                "logg_periodicity": y_periodicity,
                "teff_std": x_std,
                "logg_std": y_std,
                "n_clusters": n_clusters,
                "pattern_features": pattern_features,
                "predicted_points": {
                    "teff": pred_x.tolist(),
                    "logg": pred_y.tolist()
                }
            }
            
            json_report_file = os.path.join(
                self.report_output_dir, 
                "periodicities", 
                f"periodicity_analysis_{timestamp}.json"
            )
            
            with open(json_report_file, 'w') as f:
                json.dump(json_results, f, indent=4)
            
            return json_report_file
            
        except Exception as e:
            logger.error(f"Error in enhanced periodicity detection: {e}")
            return None
    
    def load_data(self, json_file: str) -> Dict[str, Any]:
        """Load quality data from JSON file."""
        try:
            with open(json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data from {json_file}: {e}")
            return {}
    
    def extract_grid_data(self, data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract Teff, logg, and Quality values from data."""
        teff_values = []
        logg_values = []
        quality_shifted_values = []
        z_scale_values = []

        for model_name, model_data in data.items():
            try:
                teff = float(model_data.get('teff', '0'))
                logg = float(model_data.get('logg', '0'))
                quality_shifted = float(model_data.get('Quality_shifted', 0))
                z_scale = model_data.get('z_scale', '0')
                
                teff_values.append(teff)
                logg_values.append(logg)
                quality_shifted_values.append(quality_shifted)
                z_scale_values.append(z_scale)
            except (ValueError, TypeError, KeyError) as e:
                logger.warning(f"Could not parse data for {model_name}: {e}")
        
        return np.array(teff_values), np.array(logg_values), np.array(quality_shifted_values), np.array(z_scale_values)
    
    def enhanced_periodicity_detection(self, x_values: np.ndarray, y_values: np.ndarray, 
                               z_values: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Enhanced detection of periodicities using multiple methods and confidence estimates.
        
        Args:
            x_values: Array of x-values (Teff)
            y_values: Array of y-values (logg)
            z_values: Array of z-values (quality)
            
        Returns:
            x_periodicity, y_periodicity, x_std, y_std: Detected periodicities and uncertainties
        """
        logger.info("Running enhanced periodicity detection...")
        
        # Method 1: Direct spacing analysis
        x_unique = np.sort(np.unique(x_values))
        y_unique = np.sort(np.unique(y_values))
        
        x_spacing = np.diff(x_unique) if len(x_unique) > 1 else [100]
        y_spacing = np.diff(y_unique) if len(y_unique) > 1 else [0.05]
        
        spacing_x_period = np.mean(x_spacing) if len(x_spacing) > 0 else 100
        spacing_y_period = np.mean(y_spacing) if len(y_spacing) > 0 else 0.05
        spacing_x_std = np.std(x_spacing) if len(x_spacing) > 1 else 0
        spacing_y_std = np.std(y_spacing) if len(y_spacing) > 1 else 0
        
        # Method 2: FFT-based analysis
        try:
            # Create a regular grid for FFT analysis
            xi = np.linspace(min(x_values), max(x_values), 128)
            yi = np.linspace(min(y_values), max(y_values), 128)
            zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
            
            # Replace NaN values with zeros for FFT
            zi = np.nan_to_num(zi, nan=np.nanmean(zi[~np.isnan(zi)]))
            
            # Apply windowing to reduce spectral leakage
            window = np.hanning(zi.shape[0])[:, None] * np.hanning(zi.shape[1])[None, :]
            zi_windowed = zi * window
            
            # 2D FFT analysis
            fft_2d = np.fft.fft2(zi_windowed)
            fft_2d_shifted = np.fft.fftshift(fft_2d)
            fft_2d_mag = np.abs(fft_2d_shifted)
            
            # Exclude DC component (center of the FFT)
            center_x, center_y = fft_2d_mag.shape[0] // 2, fft_2d_mag.shape[1] // 2
            fft_2d_mag[center_x-1:center_x+2, center_y-1:center_y+2] = 0
            
            # Find peaks in 2D FFT
            fft_threshold = np.max(fft_2d_mag) / 10
            peaks_y, peaks_x = np.where(fft_2d_mag > fft_threshold)
            
            # Convert FFT peak positions to frequency and then periodicity
            freq_y = (peaks_y - fft_2d_mag.shape[0] / 2) / fft_2d_mag.shape[0] * (len(yi) / (max(yi) - min(yi)))
            freq_x = (peaks_x - fft_2d_mag.shape[1] / 2) / fft_2d_mag.shape[1] * (len(xi) / (max(xi) - min(xi)))
            
            # Convert frequencies to periods and remove invalid values
            periods_x = []
            periods_y = []
            magnitudes = []
            
            for i in range(len(freq_x)):
                if freq_x[i] != 0 and freq_y[i] != 0:
                    period_x = abs(1 / freq_x[i])
                    period_y = abs(1 / freq_y[i])
                    
                    # Filter out unreasonably large or small periods
                    if (period_x < (max(x_values) - min(x_values)) and 
                        period_y < (max(y_values) - min(y_values)) and
                        period_x > 0 and period_y > 0):
                        periods_x.append(period_x)
                        periods_y.append(period_y)
                        magnitudes.append(fft_2d_mag[peaks_y[i], peaks_x[i]])
            
            # Calculate weighted average of periods based on magnitude
            fft_x_period = None
            fft_y_period = None
            fft_x_std = None
            fft_y_std = None
            
            if periods_x and periods_y:
                magnitudes = np.array(magnitudes) / np.sum(magnitudes)  # Normalize weights
                
                # Group similar periodicities to handle multiple harmonics
                grouped_x_periods = self._group_similar_values(periods_x, magnitudes)
                grouped_y_periods = self._group_similar_values(periods_y, magnitudes)
                
                # Get the most significant group for each dimension
                if grouped_x_periods:
                    best_x_group = max(grouped_x_periods, key=lambda g: g['total_weight'])
                    fft_x_period = best_x_group['weighted_mean']
                    fft_x_std = best_x_group['weighted_std']
                
                if grouped_y_periods:
                    best_y_group = max(grouped_y_periods, key=lambda g: g['total_weight'])
                    fft_y_period = best_y_group['weighted_mean']
                    fft_y_std = best_y_group['weighted_std']
        except Exception as e:
            logger.warning(f"FFT-based periodicity detection failed: {e}")
            fft_x_period = None
            fft_y_period = None
            fft_x_std = None
            fft_y_std = None
        
        # Method 3: 1D projections and autocorrelation
        try:
            # Get 1D projections by averaging across axes
            x_projection = np.nanmean(zi, axis=0)
            y_projection = np.nanmean(zi, axis=1)
            
            # Compute autocorrelation
            x_autocorr = np.correlate(x_projection, x_projection, mode='full')
            y_autocorr = np.correlate(y_projection, y_projection, mode='full')
            
            # Keep only the positive lag part (second half)
            x_autocorr = x_autocorr[len(x_autocorr)//2:]
            y_autocorr = y_autocorr[len(y_autocorr)//2:]
            
            # Find peaks in autocorrelation
            x_peaks, _ = signal.find_peaks(x_autocorr, height=np.max(x_autocorr)/3)
            y_peaks, _ = signal.find_peaks(y_autocorr, height=np.max(y_autocorr)/3)
            
            # Convert peak lags to periods
            autocorr_x_periods = []
            autocorr_y_periods = []
            
            if len(x_peaks) > 0:
                x_lags = x_peaks[x_peaks > 0]  # Skip the zero lag
                if len(x_lags) > 0:
                    x_lag = x_lags[0]  # First peak is likely the primary periodicity
                    autocorr_x_periods.append((max(xi) - min(xi)) * x_lag / len(xi))
            
            if len(y_peaks) > 0:
                y_lags = y_peaks[y_peaks > 0]  # Skip the zero lag
                if len(y_lags) > 0:
                    y_lag = y_lags[0]  # First peak is likely the primary periodicity
                    autocorr_y_periods.append((max(yi) - min(yi)) * y_lag / len(yi))
            
            autocorr_x_period = np.mean(autocorr_x_periods) if autocorr_x_periods else None
            autocorr_y_period = np.mean(autocorr_y_periods) if autocorr_y_periods else None
            autocorr_x_std = np.std(autocorr_x_periods) if len(autocorr_x_periods) > 1 else spacing_x_std
            autocorr_y_std = np.std(autocorr_y_periods) if len(autocorr_y_periods) > 1 else spacing_y_std
            
        except Exception as e:
            logger.warning(f"Autocorrelation-based periodicity detection failed: {e}")
            autocorr_x_period = None
            autocorr_y_period = None
            autocorr_x_std = None
            autocorr_y_std = None
        
        # Combine results from different methods
        x_periods = []
        y_periods = []
        x_weights = []
        y_weights = []
        
        # Add spacing-based result
        x_periods.append(spacing_x_period)
        y_periods.append(spacing_y_period)
        x_weights.append(1.0)  # Default weight
        y_weights.append(1.0)
        
        # Add FFT-based result if available
        if fft_x_period is not None and np.isfinite(fft_x_period):
            x_periods.append(fft_x_period)
            x_weights.append(2.0)  # Higher weight for FFT result
        
        if fft_y_period is not None and np.isfinite(fft_y_period):
            y_periods.append(fft_y_period)
            y_weights.append(2.0)
        
        # Add autocorrelation result if available
        if autocorr_x_period is not None and np.isfinite(autocorr_x_period):
            x_periods.append(autocorr_x_period)
            x_weights.append(1.5)  # Medium weight for autocorrelation
        
        if autocorr_y_period is not None and np.isfinite(autocorr_y_period):
            y_periods.append(autocorr_y_period)
            y_weights.append(1.5)
        
        # Calculate weighted means
        x_weights = np.array(x_weights) / np.sum(x_weights)
        y_weights = np.array(y_weights) / np.sum(y_weights)
        
        x_periodicity = np.sum(np.array(x_periods) * x_weights)
        y_periodicity = np.sum(np.array(y_periods) * y_weights)
        
        # Calculate weighted standard deviations
        if len(x_periods) > 1:
            x_std = np.sqrt(np.sum(x_weights * (np.array(x_periods) - x_periodicity)**2))
        else:
            x_std = spacing_x_std
            
        if len(y_periods) > 1:
            y_std = np.sqrt(np.sum(y_weights * (np.array(y_periods) - y_periodicity)**2))
        else:
            y_std = spacing_y_std
        
        # Validate results
        if not np.isfinite(x_periodicity) or x_periodicity <= 0:
            logger.warning("Invalid x periodicity detected, using spacing-based default")
            x_periodicity = spacing_x_period
            x_std = spacing_x_std
            
        if not np.isfinite(y_periodicity) or y_periodicity <= 0:
            logger.warning("Invalid y periodicity detected, using spacing-based default")
            y_periodicity = spacing_y_period
            y_std = spacing_y_std
        
        logger.info(f"Enhanced periodicity detection complete: ")
        logger.info(f"  Teff periodicity: {x_periodicity:.2f} ± {x_std:.2f}")
        logger.info(f"  logg periodicity: {y_periodicity:.4f} ± {y_std:.4f}")
        logger.info(f"  Methods used: {len(x_periods)} for Teff, {len(y_periods)} for logg")
        
        return x_periodicity, y_periodicity, x_std, y_std
    
    def _group_similar_values(self, values, weights, similarity_threshold=0.2):
        """
        Group similar values together for more robust periodicity detection.
        
        Args:
            values: List of values to group
            weights: Weights for each value
            similarity_threshold: Threshold for considering values similar
            
        Returns:
            List of dictionaries with group statistics
        """
        if not values:
            return []
            
        groups = []
        values = np.array(values)
        weights = np.array(weights)
        
        # Sort by weight descending so we start with most significant values
        sorted_indices = np.argsort(-weights)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        for i, (value, weight) in enumerate(zip(sorted_values, sorted_weights)):
            # Check if this value fits in any existing group
            found_group = False
            for group in groups:
                relative_diff = abs(value - group['weighted_mean']) / group['weighted_mean']
                if relative_diff < similarity_threshold:
                    # Add to group
                    group['values'].append(value)
                    group['weights'].append(weight)
                    
                    # Update group statistics
                    group['total_weight'] += weight
                    group['weighted_mean'] = np.average(group['values'], weights=group['weights'])
                    group['weighted_std'] = np.sqrt(np.average(
                        (np.array(group['values']) - group['weighted_mean'])**2, 
                        weights=group['weights']
                    ))
                    found_group = True
                    break
            
            if not found_group:
                # Create new group
                groups.append({
                    'values': [value],
                    'weights': [weight],
                    'total_weight': weight,
                    'weighted_mean': value,
                    'weighted_std': 0
                })
        
        return groups
    
    def enhanced_pattern_detection(self, x_values: np.ndarray, y_values: np.ndarray, 
                               z_values: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Detect patterns in the data using enhanced clustering techniques.
        
        Args:
            x_values: Array of x-values (Teff)
            y_values: Array of y-values (logg)
            z_values: Array of z-values (quality)
            
        Returns:
            cluster_labels: Array of cluster labels
            n_clusters: Number of clusters found
        """
        logger.info("Running enhanced pattern detection...")
        
        # Normalize the data for clustering
        x_norm = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
        y_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
        z_norm = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values) + 1e-10)
        
        # Prepare data for clustering with emphasis on quality values
        data = np.column_stack([
            x_norm,
            y_norm,
            z_norm * 2  # Higher weight for quality
        ])
        
        # Try DBSCAN with adaptive parameters
        try:
            # Estimate optimal epsilon based on data density
            from sklearn.neighbors import NearestNeighbors
            nbrs = NearestNeighbors(n_neighbors=min(5, len(x_values)-1)).fit(data)
            distances, _ = nbrs.kneighbors(data)
            
            # Sort and find "elbow" in distance graph for epsilon selection
            dist_sorted = np.sort(distances[:, -1])
            
            # Estimate the "elbow" by looking at second derivatives
            dists = np.diff(dist_sorted)
            dists2 = np.diff(dists)
            elbow_idx = np.argmax(dists2) + 2 if len(dists2) > 0 else len(dist_sorted) // 3
            adaptive_eps = dist_sorted[min(elbow_idx, len(dist_sorted)-1)]
            
            # Adjust eps based on data size 
            eps = min(adaptive_eps, 0.2) if len(x_values) > 20 else 0.3
            min_samples = max(2, min(5, len(x_values) // 10))
            
            # Run DBSCAN
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(data)
            labels_dbscan = clustering.labels_
            n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
            
            logger.info(f"DBSCAN found {n_clusters_dbscan} clusters with eps={eps:.3f}")
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")
            labels_dbscan = np.ones(len(x_values), dtype=int) * -1
            n_clusters_dbscan = 0
        
        # Try KMeans clustering
        try:
            # Use silhouette analysis to determine optimal number of clusters
            from sklearn.metrics import silhouette_score
            
            # Try a range of cluster numbers
            max_clusters = min(len(x_values) // 3, 8)
            silhouette_scores = []
            
            for n_clusters in range(2, max_clusters + 1):
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(data)
                
                if len(np.unique(cluster_labels)) > 1:  # Need at least 2 clusters for silhouette
                    silhouette_avg = silhouette_score(data, cluster_labels)
                    silhouette_scores.append((n_clusters, silhouette_avg))
            
            # Find optimal number of clusters
            if silhouette_scores:
                optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
                
                # Run KMeans with optimal clusters
                kmeans = KMeans(n_clusters=optimal_n_clusters, random_state=42, n_init=10)
                labels_kmeans = kmeans.fit_predict(data)
                n_clusters_kmeans = optimal_n_clusters
                
                logger.info(f"KMeans found {n_clusters_kmeans} clusters")
            else:
                labels_kmeans = np.ones(len(x_values), dtype=int) * -1
                n_clusters_kmeans = 0
                logger.warning("KMeans clustering failed - could not determine optimal clusters")
        except Exception as e:
            logger.warning(f"KMeans clustering failed: {e}")
            labels_kmeans = np.ones(len(x_values), dtype=int) * -1
            n_clusters_kmeans = 0
        
        # Choose the better clustering result
        if n_clusters_dbscan > 0 and n_clusters_kmeans > 0:
            # If both methods found clusters, use the one with better silhouette score
            try:
                if n_clusters_dbscan > 1 and len(np.unique(labels_dbscan)) > 1:
                    silhouette_dbscan = silhouette_score(data, labels_dbscan)
                else:
                    silhouette_dbscan = -1
                    
                if n_clusters_kmeans > 1 and len(np.unique(labels_kmeans)) > 1:
                    silhouette_kmeans = silhouette_score(data, labels_kmeans)
                else:
                    silhouette_kmeans = -1
                
                if silhouette_dbscan >= silhouette_kmeans:
                    logger.info(f"Using DBSCAN clustering (score: {silhouette_dbscan:.3f})")
                    cluster_labels = labels_dbscan
                    n_clusters = n_clusters_dbscan
                else:
                    logger.info(f"Using KMeans clustering (score: {silhouette_kmeans:.3f})")
                    cluster_labels = labels_kmeans
                    n_clusters = n_clusters_kmeans
            except Exception as e:
                logger.warning(f"Error comparing clustering methods: {e}, using DBSCAN")
                cluster_labels = labels_dbscan
                n_clusters = n_clusters_dbscan
        elif n_clusters_dbscan > 0:
            cluster_labels = labels_dbscan
            n_clusters = n_clusters_dbscan
        elif n_clusters_kmeans > 0:
            cluster_labels = labels_kmeans
            n_clusters = n_clusters_kmeans
        else:
            # If no clusters found, try gradient-based segmentation
            logger.info("No clusters found with standard methods, trying gradient-based segmentation")
            try:
                # Create a regular grid for gradient calculation
                xi = np.linspace(min(x_values), max(x_values), 50)
                yi = np.linspace(min(y_values), max(y_values), 50)
                zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
                zi = np.nan_to_num(zi, nan=np.nanmean(zi[~np.isnan(zi)]))
                
                # Calculate gradients
                gradient_y, gradient_x = np.gradient(zi)
                gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                
                # Threshold gradient to find edges
                threshold = np.percentile(gradient_magnitude, 75)
                edges = gradient_magnitude > threshold
                
                # Label connected regions
                from scipy import ndimage
                labeled_regions, n_regions = ndimage.label(~edges)
                
                # Map grid regions back to original points
                region_labels = np.zeros_like(x_values, dtype=int)
                for i, (x, y) in enumerate(zip(x_values, y_values)):
                    # Find nearest grid point
                    x_idx = np.argmin(np.abs(xi - x))
                    y_idx = np.argmin(np.abs(yi - y))
                    region_labels[i] = labeled_regions[y_idx, x_idx] - 1  # -1 to match clustering convention
                
                if n_regions > 1:
                    cluster_labels = region_labels
                    n_clusters = n_regions
                    logger.info(f"Gradient-based segmentation found {n_regions} regions")
                else:
                    # Last resort: manual thresholding on quality
                    threshold = np.percentile(z_values, 50)
                    cluster_labels = (z_values <= threshold).astype(int)
                    n_clusters = 2
                    logger.info("Using simple quality thresholding: 2 regions")
            except Exception as e:
                logger.warning(f"Gradient-based segmentation failed: {e}")
                cluster_labels = np.zeros_like(x_values, dtype=int)
                n_clusters = 1
                logger.info("Fallback to single cluster")
        
        logger.info(f"Enhanced pattern detection found {n_clusters} clusters/patterns")
        return cluster_labels, n_clusters
    
    def extract_pattern_features(self, x_values: np.ndarray, y_values: np.ndarray, 
                              z_values: np.ndarray) -> Dict[str, Any]:
        """
        Extract detailed pattern features from the data for enhanced analysis.
        
        Args:
            x_values: Array of x-values (Teff)
            y_values: Array of y-values (logg)
            z_values: Array of z-values (quality)
            
        Returns:
            Dictionary with pattern features
        """
        pattern_features = {}
        
        try:
            # Create a regular grid for gradient calculation
            xi = np.linspace(min(x_values), max(x_values), 50)
            yi = np.linspace(min(y_values), max(y_values), 50)
            zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
            zi = np.nan_to_num(zi, nan=np.nanmean(zi[~np.isnan(zi)]))
            
            # Calculate gradients
            gradient_y, gradient_x = np.gradient(zi)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Find high gradient regions
            high_gradient_threshold = np.percentile(gradient_magnitude, 80)
            high_gradient_mask = gradient_magnitude > high_gradient_threshold
            
            # Extract regions with high gradients
            high_gradient_regions = []
            if np.any(high_gradient_mask):
                from scipy import ndimage
                labeled_regions, n_regions = ndimage.label(high_gradient_mask)
                
                for region_idx in range(1, n_regions + 1):
                    region_mask = labeled_regions == region_idx
                    if np.sum(region_mask) > 4:  # Ignore very small regions
                        y_indices, x_indices = np.where(region_mask)
                        x_min = xi[np.min(x_indices)]
                        x_max = xi[np.max(x_indices)]
                        y_min = yi[np.min(y_indices)]
                        y_max = yi[np.max(y_indices)]
                        high_gradient_regions.append((x_min, x_max, y_min, y_max))
            
            pattern_features['quality_gradient_regions'] = high_gradient_regions
            
            # Calculate sensitivity of quality to Teff and logg changes
            if len(x_values) >= 5:
                # For Teff sensitivity, find points with similar logg
                teff_sensitivities = []
                for logg in np.unique(y_values):
                    mask = np.isclose(y_values, logg, atol=0.01)
                    if np.sum(mask) >= 2:
                        teff_subset = x_values[mask]
                        quality_subset = z_values[mask]
                        
                        # Sort by Teff
                        sort_idx = np.argsort(teff_subset)
                        teff_sorted = teff_subset[sort_idx]
                        quality_sorted = quality_subset[sort_idx]
                        
                        # Calculate average change in quality per 100K change in Teff
                        diffs = np.diff(quality_sorted) / (np.diff(teff_sorted) / 100)
                        if len(diffs) > 0:
                            teff_sensitivities.extend(diffs)
                
                # For logg sensitivity, find points with similar Teff
                logg_sensitivities = []
                for teff in np.unique(x_values):
                    mask = np.isclose(x_values, teff, atol=50)
                    if np.sum(mask) >= 2:
                        logg_subset = y_values[mask]
                        quality_subset = z_values[mask]
                        
                        # Sort by logg
                        sort_idx = np.argsort(logg_subset)
                        logg_sorted = logg_subset[sort_idx]
                        quality_sorted = quality_subset[sort_idx]
                        
                        # Calculate average change in quality per 0.1 change in logg
                        diffs = np.diff(quality_sorted) / (np.diff(logg_sorted) / 0.1)
                        if len(diffs) > 0:
                            logg_sensitivities.extend(diffs)
                
                # Calculate average sensitivities
                if teff_sensitivities:
                    pattern_features['teff_sensitivity'] = np.median(teff_sensitivities)
                if logg_sensitivities:
                    pattern_features['logg_sensitivity'] = np.median(logg_sensitivities)
            
            # Calculate pattern symmetry
            try:
                # Calculate symmetry with respect to the center of the data
                center_x = (max(x_values) + min(x_values)) / 2
                center_y = (max(y_values) + min(y_values)) / 2
                
                # Create grid for symmetry calculation
                xi_sym = np.linspace(min(x_values), max(x_values), 30)
                yi_sym = np.linspace(min(y_values), max(y_values), 30)
                zi_sym = griddata((x_values, y_values), z_values, (xi_sym[None,:], yi_sym[:,None]), method='cubic')
                zi_sym = np.nan_to_num(zi_sym, nan=np.nanmean(zi_sym[~np.isnan(zi_sym)]))
                
                # Mirror the grid in x and y directions
                zi_sym_x = np.zeros_like(zi_sym)
                zi_sym_y = np.zeros_like(zi_sym)
                
                for i in range(zi_sym.shape[0]):
                    # Mirror in x-direction (left-right)
                    zi_sym_x[i,:] = zi_sym[i,::-1]
                
                for j in range(zi_sym.shape[1]):
                    # Mirror in y-direction (up-down)
                    zi_sym_y[:,j] = zi_sym[::-1,j]
                
                # Calculate symmetry scores (1 - normalized difference)
                x_sym_diff = np.sum(np.abs(zi_sym - zi_sym_x)) / np.sum(np.abs(zi_sym))
                y_sym_diff = np.sum(np.abs(zi_sym - zi_sym_y)) / np.sum(np.abs(zi_sym))
                
                x_sym_score = 1 - min(x_sym_diff, 1.0)
                y_sym_score = 1 - min(y_sym_diff, 1.0)
                
                # Overall symmetry (average of x and y)
                symmetry_score = (x_sym_score + y_sym_score) / 2
                pattern_features['symmetry_score'] = symmetry_score
                pattern_features['symmetry_x'] = x_sym_score
                pattern_features['symmetry_y'] = y_sym_score
            except Exception as e:
                logger.warning(f"Symmetry calculation failed: {e}")
            
            # Detect local minima in quality
            try:
                from scipy import ndimage
                
                # Find local minima in the interpolated grid
                local_min = (zi == ndimage.minimum_filter(zi, size=3))
                
                minima_values = []
                minima_positions = []
                
                if np.any(local_min):
                    y_min_indices, x_min_indices = np.where(local_min)
                    
                    for x_idx, y_idx in zip(x_min_indices, y_min_indices):
                        if 0 < x_idx < len(xi) - 1 and 0 < y_idx < len(yi) - 1:
                            minima_positions.append((xi[x_idx], yi[y_idx]))
                            minima_values.append(zi[y_idx, x_idx])
                
                if minima_positions:
                    # Sort by quality (ascending)
                    sort_idx = np.argsort(minima_values)
                    pattern_features['local_minima'] = [
                        (float(pos[0]), float(pos[1]), float(val))
                        for pos, val in zip(
                            [minima_positions[i] for i in sort_idx],
                            [minima_values[i] for i in sort_idx]
                        )
                    ]
            except Exception as e:
                logger.warning(f"Local minima detection failed: {e}")
            
        except Exception as e:
            logger.warning(f"Error extracting pattern features: {e}")
        
        return pattern_features
    
    def pattern_based_prediction(self, x_values: np.ndarray, y_values: np.ndarray, 
                              z_values: np.ndarray, x_periodicity: float, y_periodicity: float,
                              pattern_features: Dict[str, Any], x_range: Tuple[float, float],
                              y_range: Tuple[float, float], n_points: int = 200) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions using pattern-aware approach for optimal sampling.
        
        Args:
            x_values: Original x-values (Teff)
            y_values: Original y-values (logg)
            z_values: Original z-values (quality)
            x_periodicity, y_periodicity: Detected periodicities
            pattern_features: Dictionary with pattern features
            x_range, y_range: Range constraints
            n_points: Number of points to predict
            
        Returns:
            pred_x, pred_y: Predicted point coordinates
        """
        logger.info(f"Generating {n_points} predictions using pattern-based approach...")
        
        # Set exploration boundaries
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        try:
            # Create a grid for analysis
            grid_size_x = max(30, int(np.ceil((x_max - x_min) / (x_periodicity / 4))))
            grid_size_y = max(30, int(np.ceil((y_max - y_min) / (y_periodicity / 4))))
            
            grid_x = np.linspace(x_min, x_max, grid_size_x)
            grid_y = np.linspace(y_min, y_max, grid_size_y)
            XX, YY = np.meshgrid(grid_x, grid_y)
            grid_points_x = XX.flatten()
            grid_points_y = YY.flatten()
            
            # 1. Calculate multiple scoring criteria for all grid points
            
            # Distance from existing points (larger is better for exploration)
            min_distances = np.zeros(len(grid_points_x))
            for i in range(len(grid_points_x)):
                # Calculate distances to all existing points
                distances = np.sqrt((grid_points_x[i] - x_values)**2 + 
                                  (grid_points_y[i] - y_values)**2)
                min_distances[i] = np.min(distances)
            
            # Normalize to [0, 1] range
            if np.max(min_distances) > np.min(min_distances):
                distance_scores = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))
            else:
                distance_scores = np.ones_like(min_distances) * 0.5
            
            # 2. Calculate grid point adherence to detected periodicity
            periodicity_scores = np.zeros(len(grid_points_x))
            
            # Calculate coordinates relative to the center of the data
            center_x = (max(x_values) + min(x_values)) / 2
            center_y = (max(y_values) + min(y_values)) / 2
            
            for i in range(len(grid_points_x)):
                # X periodicity score - how close to being on the grid
                x_phase = ((grid_points_x[i] - center_x) % x_periodicity) / x_periodicity
                x_score = 1.0 - min(x_phase, 1 - x_phase) * 2  # Higher when closer to grid lines
                
                # Y periodicity score
                y_phase = ((grid_points_y[i] - center_y) % y_periodicity) / y_periodicity
                y_score = 1.0 - min(y_phase, 1 - y_phase) * 2
                
                # Combined periodicity score
                periodicity_scores[i] = (x_score + y_score) / 2
            
            # 3. Calculate gradient exploration scores
            gradient_scores = np.zeros(len(grid_points_x))
            
            if pattern_features.get('quality_gradient_regions'):
                # Assign high scores to points in high gradient regions
                for region in pattern_features['quality_gradient_regions']:
                    x_min_reg, x_max_reg, y_min_reg, y_max_reg = region
                    # Find points in this region
                    in_region = ((grid_points_x >= x_min_reg) & (grid_points_x <= x_max_reg) &
                               (grid_points_y >= y_min_reg) & (grid_points_y <= y_max_reg))
                    # Assign high score to these points
                    gradient_scores[in_region] = 1.0
            
            # If no gradient regions identified, use interpolated gradient
            if np.max(gradient_scores) == 0:
                try:
                    # Create a regular grid for gradient calculation
                    xi = np.linspace(min(x_values), max(x_values), 50)
                    yi = np.linspace(min(y_values), max(y_values), 50)
                    zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
                    zi = np.nan_to_num(zi, nan=np.nanmean(zi[~np.isnan(zi)]))
                    
                    # Calculate gradients
                    gradient_y, gradient_x = np.gradient(zi)
                    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
                    
                    # Interpolate gradient magnitude to grid points
                    for i in range(len(grid_points_x)):
                        if min(x_values) <= grid_points_x[i] <= max(x_values) and min(y_values) <= grid_points_y[i] <= max(y_values):
                            # Find closest grid point in the interpolated grid
                            x_idx = np.argmin(np.abs(xi - grid_points_x[i]))
                            y_idx = np.argmin(np.abs(yi - grid_points_y[i]))
                            gradient_scores[i] = gradient_magnitude[y_idx, x_idx]
                    
                    # Normalize gradient scores
                    if np.max(gradient_scores) > 0:
                        gradient_scores = gradient_scores / np.max(gradient_scores)
                except Exception as e:
                    logger.warning(f"Gradient score calculation failed: {e}")
            
            # 4. Calculate scores based on local minima exploration
            minima_scores = np.zeros(len(grid_points_x))
            
            if pattern_features.get('local_minima'):
                for x_min, y_min, _ in pattern_features['local_minima']:
                    # Assign higher scores to points in the vicinity of local minima
                    for i in range(len(grid_points_x)):
                        # Calculate distance to this minimum
                        dist = np.sqrt((grid_points_x[i] - x_min)**2 + (grid_points_y[i] - y_min)**2)
                        # Score decreases with distance, up to a certain radius
                        radius = min(x_periodicity, y_periodicity * 100) / 2
                        if dist < radius:
                            score = 1.0 - (dist / radius)
                            minima_scores[i] = max(minima_scores[i], score)
            
            # 5. Combine all scoring criteria with weights
            combined_scores = (
                0.3 * distance_scores +      # Exploration
                0.3 * periodicity_scores +   # Pattern adherence
                0.3 * gradient_scores +      # Gradient exploration
                0.1 * minima_scores          # Minima exploration
            )
            
            # Filter points that are too close to existing data
            min_allowed_distance = min(x_periodicity / 8, y_periodicity * 50)
            valid_points_mask = min_distances > min_allowed_distance
            
            if np.sum(valid_points_mask) >= n_points:
                filtered_x = grid_points_x[valid_points_mask]
                filtered_y = grid_points_y[valid_points_mask]
                filtered_scores = combined_scores[valid_points_mask]
                
                # Select top n_points based on scores
                top_indices = np.argsort(filtered_scores)[-n_points:]
                pred_x = filtered_x[top_indices]
                pred_y = filtered_y[top_indices]
            else:
                # Use all valid points and relax distance criteria for the rest
                filtered_x = grid_points_x[valid_points_mask]
                filtered_y = grid_points_y[valid_points_mask]
                filtered_scores = combined_scores[valid_points_mask]
                
                # Get remaining points needed
                remaining_n = n_points - len(filtered_x)
                
                # Sort the invalid (too close) points by score 
                invalid_mask = ~valid_points_mask
                remaining_x = grid_points_x[invalid_mask]
                remaining_y = grid_points_y[invalid_mask]
                remaining_scores = combined_scores[invalid_mask]
                
                # Get top remaining_n points
                if len(remaining_x) > remaining_n:
                    remaining_top_indices = np.argsort(remaining_scores)[-remaining_n:]
                    remaining_pred_x = remaining_x[remaining_top_indices]
                    remaining_pred_y = remaining_y[remaining_top_indices]
                else:
                    remaining_pred_x = remaining_x
                    remaining_pred_y = remaining_y
                
                # Combine all points
                pred_x = np.concatenate([filtered_x, remaining_pred_x])
                pred_y = np.concatenate([filtered_y, remaining_pred_y])
            
            logger.info(f"Generated {len(pred_x)} predictions using pattern-based approach")
            return pred_x, pred_y
            
        except Exception as e:
            logger.error(f"Pattern-based prediction failed: {e}")
            logger.info("Falling back to simple grid sampling")
            
            # Simple grid fallback
            try:
                grid_x = np.linspace(x_min, x_max, 20)
                grid_y = np.linspace(y_min, y_max, 20)
                XX, YY = np.meshgrid(grid_x, grid_y)
                candidates_x = XX.flatten()
                candidates_y = YY.flatten()
                
                # Filter out points too close to existing data
                min_distances = []
                for i in range(len(candidates_x)):
                    distances = np.sqrt((candidates_x[i] - x_values)**2 + 
                                     (candidates_y[i] - y_values)**2)
                    min_distances.append(np.min(distances))
                
                min_distances = np.array(min_distances)
                
                # Find points with sufficient distance
                valid_indices = np.where(min_distances > min(x_periodicity / 8, y_periodicity * 50))[0]
                
                if len(valid_indices) >= n_points:
                    selected_indices = valid_indices[:n_points]
                    pred_x = candidates_x[selected_indices]
                    pred_y = candidates_y[selected_indices]
                else:
                    pred_x = candidates_x[valid_indices]
                    pred_y = candidates_y[valid_indices]
                
                logger.info(f"Generated {len(pred_x)} predictions using simple grid fallback")
                return pred_x, pred_y
            except Exception as e2:
                logger.error(f"Simple grid fallback failed: {e2}")
                # Return empty arrays as last resort
                return np.array([]), np.array([])
    
    def enhanced_visualization(self, x_values: np.ndarray, y_values: np.ndarray, z_values: np.ndarray,
                    pred_x: np.ndarray, pred_y: np.ndarray, x_periodicity: float, 
                    y_periodicity: float, cluster_labels: np.ndarray, n_clusters: int, 
                    pattern_features: Dict[str, Any], z_scale: str, output_file: str, 
                    x_range: Optional[Tuple[float, float]] = None, 
                    y_range: Optional[Tuple[float, float]] = None):
        """
        Create enhanced visualization of periodicity analysis with pattern highlighting.
        
        Args:
            x_values: Array of x-values (Teff)
            y_values: Array of y-values (logg)
            z_values: Array of z-values (quality)
            pred_x: Array of predicted x coordinates
            pred_y: Array of predicted y coordinates
            x_periodicity: Detected x periodicity
            y_periodicity: Detected y periodicity
            cluster_labels: Array of cluster labels
            n_clusters: Number of clusters
            pattern_features: Dictionary with pattern features
            z_scale: Z-scale value
            output_file: Path to save the output image
            x_range, y_range: Range for the plot
        """
        plt.figure(figsize=(14, 12))
        
        # Create a custom colormap
        colors = ['darkviolet', 'navy', 'teal', 'green', 'yellowgreen', 'yellow']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Calculate bounds for the plot
        if len(pred_x) > 0:
            x_min, x_max = min(min(x_values), min(pred_x)), max(max(x_values), max(pred_x))
            y_min, y_max = min(min(y_values), min(pred_y)), max(max(y_values), max(pred_y))
        else:
            x_min, x_max = min(x_values), max(x_values)
            y_min, y_max = min(y_values), max(y_values)
        
        # Apply ranges if specified
        if x_range:
            x_min, x_max = x_range
        if y_range:
            y_min, y_max = y_range
        
        # Add some padding
        x_range_size = x_max - x_min
        y_range_size = y_max - y_min
        x_min -= 0.05 * x_range_size
        x_max += 0.05 * x_range_size
        y_min -= 0.05 * y_range_size
        y_max += 0.05 * y_range_size
        
        # Create a grid for the contour plot (using more points for smoother contours)
        xi = np.linspace(min(x_values), max(x_values), 150)
        yi = np.linspace(min(y_values), max(y_values), 150)
        zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
        
        # Replace NaN values with the mean of non-NaN values
        if np.any(np.isnan(zi)):
            zi = np.nan_to_num(zi, nan=np.nanmean(zi[~np.isnan(zi)]))
        
        # Apply a slight Gaussian filter for smoother contours
        zi = ndimage.gaussian_filter(zi, sigma=1.0)
        
        # Plot the contour
        contour = plt.contourf(xi, yi, zi, 100, cmap=cmap, extend='both')
        cb = plt.colorbar(contour, label='Quality (Shifted)')
        cb.ax.tick_params(labelsize=12)
        
        # Calculate gradients for highlighting important regions
        gradient_y, gradient_x = np.gradient(zi)
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Highlight high gradient areas
        high_gradient_threshold = np.percentile(gradient_magnitude, 85)
        high_gradient_mask = gradient_magnitude > high_gradient_threshold
        
        if np.any(high_gradient_mask):
            # Plot contour lines for high gradient areas
            gradient_contour = plt.contour(xi, yi, gradient_magnitude, 
                                        levels=[high_gradient_threshold], 
                                        colors='black', linestyles='dotted', 
                                        linewidths=1.5, alpha=0.7)
            plt.clabel(gradient_contour, inline=True, fontsize=10, fmt='High Gradient')
        
        # Plot the original data points
        scatter = plt.scatter(x_values, y_values, marker='o', s=80, 
                           c=z_values, cmap='plasma',
                           edgecolor='white', linewidth=1.5,
                           label='Original Data Points')
        
        # Highlight detected clusters if they exist
        unique_labels = set(cluster_labels)
        if n_clusters > 0:  # If we have more than just noise (-1)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                mask = cluster_labels == label
                plt.scatter(x_values[mask], y_values[mask], marker='*', s=200,
                          edgecolor='white', linewidth=1.5, 
                          label=f'Cluster {label}')
                
                # Add convex hull around each cluster
                try:
                    from scipy.spatial import ConvexHull
                    cluster_points = np.column_stack([x_values[mask], y_values[mask]])
                    if len(cluster_points) >= 3:  # Need at least 3 points for a hull
                        hull = ConvexHull(cluster_points)
                        hull_x = cluster_points[hull.vertices,0]
                        hull_y = cluster_points[hull.vertices,1]
                        plt.fill(hull_x, hull_y, alpha=0.2, edgecolor='k', linewidth=1.5)
                except Exception as e:
                    logger.warning(f"Convex hull calculation failed for cluster {label}: {e}")
        
        # Highlight local minima if available
        if pattern_features.get('local_minima'):
            local_min_x = [p[0] for p in pattern_features['local_minima']]
            local_min_y = [p[1] for p in pattern_features['local_minima']]
            
            plt.scatter(local_min_x, local_min_y, marker='v', s=150, color='lime',
                      edgecolor='black', linewidth=1.5, label='Local Minima')
        
        # Visualize predicted points with custom density heatmap
        if len(pred_x) > 0:
            # Create a hexbin for showing prediction density
            hex_bins = plt.hexbin(pred_x, pred_y, cmap='Reds', alpha=0.3, 
                              gridsize=(15, 12), mincnt=1)
            
            # Add a sample of the predicted points
            sample_size = min(50, len(pred_x))
            sample_indices = np.random.choice(len(pred_x), sample_size, replace=False)
            plt.scatter(pred_x[sample_indices], pred_y[sample_indices], 
                      c='red', marker='x', s=50, 
                      label='Predicted Points (Sample)')
            
            # Add contour lines to show prediction density
            plt.colorbar(hex_bins, label='Prediction Density')
        
        # Add grid lines showing periodicity
        x_center = (min(x_values) + max(x_values)) / 2
        y_center = (min(y_values) + max(y_values)) / 2
        
        # Vertical lines for x periodicity
        for i in range(-10, 11):
            x_line = x_center + i * x_periodicity
            if x_min <= x_line <= x_max:
                plt.axvline(x_line, color='blue', linestyle='--', alpha=0.3)
        
        # Horizontal lines for y periodicity
        for i in range(-10, 11):
            y_line = y_center + i * y_periodicity
            if y_min <= y_line <= y_max:
                plt.axhline(y_line, color='blue', linestyle='--', alpha=0.3)
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Highlight high gradient regions with annotations
        if pattern_features.get('quality_gradient_regions'):
            for i, region in enumerate(pattern_features['quality_gradient_regions'][:3]):  # Show top 3
                x_min_reg, x_max_reg, y_min_reg, y_max_reg = region
                rect = plt.Rectangle((x_min_reg, y_min_reg), 
                                  x_max_reg - x_min_reg, y_max_reg - y_min_reg,
                                  fill=False, edgecolor='yellow', linewidth=2, linestyle='-.')
                plt.gca().add_patch(rect)
                plt.annotate(f"High Gradient Region {i+1}", 
                           xy=((x_min_reg + x_max_reg)/2, (y_min_reg + y_max_reg)/2),
                           xytext=(20, 20), textcoords='offset points',
                           fontsize=10, color='yellow',
                           arrowprops=dict(arrowstyle="->", color='yellow'))
        
        # Add sensitivity info if available
        if pattern_features.get('teff_sensitivity') and pattern_features.get('logg_sensitivity'):
            teff_sens = pattern_features['teff_sensitivity']
            logg_sens = pattern_features['logg_sensitivity']
            
            sensitivity_text = f"Quality Sensitivity:\nTeff: {teff_sens:.6f} per 100K\nlogg: {logg_sens:.6f} per 0.1 dex"
            plt.annotate(sensitivity_text, xy=(0.02, 0.12), xycoords='axes fraction',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Set labels and title
        plt.xlabel('Effective Temperature (Teff)', fontsize=14)
        plt.ylabel('Surface Gravity (log g)', fontsize=14)
        plt.title(f'Enhanced Periodicity Analysis for Stellar Atmosphere Models (z_scale={z_scale})\n'
                f'Detected Periodicities: Teff={x_periodicity:.2f}, logg={y_periodicity:.4f}', 
                fontsize=16)
        
        # Add text box with analysis info
        info_str = f'Detected Teff Periodicity: {x_periodicity:.2f}\n'
        info_str += f'Detected logg Periodicity: {y_periodicity:.4f}\n'
        info_str += f'Number of Pattern Clusters: {n_clusters}\n'
        info_str += f'Suggested Points: {len(pred_x)}'
        
        if pattern_features.get('symmetry_score'):
            info_str += f'\nPattern Symmetry: {pattern_features["symmetry_score"]:.2f}'
        
        plt.annotate(info_str, xy=(0.02, 0.02), xycoords='axes fraction',
                   fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add legend with smaller font to save space
        plt.legend(loc='upper right', fontsize=10, framealpha=0.8)
        
        # Set axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced periodicity analysis plot saved to {output_file}")