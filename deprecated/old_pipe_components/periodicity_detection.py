#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy import ndimage
from scipy.interpolate import griddata
import argparse
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import DBSCAN
import os
from scipy.fft import fft, fftfreq
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel

def load_data(json_file):
    """Load data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def extract_grid_data(data):
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
            z_scale = float(model_data.get('z_scale', 0))
            
            teff_values.append(teff)
            logg_values.append(logg)
            quality_shifted_values.append(quality_shifted)
            z_scale_values.append(z_scale)
        except (ValueError, TypeError):
            print(f"Warning: Could not parse data for {model_name}")
    
    return np.array(teff_values), np.array(logg_values), np.array(quality_shifted_values), np.array(z_scale_values)

def detect_periodicities(x_values, y_values, z_values):
    """Detect periodicities in the data using spectral analysis."""
    # For X axis (Teff)
    x_unique = np.unique(x_values)
    if len(x_unique) > 5:  # Need enough points for meaningful FFT
        x_spacing = np.diff(np.sort(x_unique))
        x_period = np.mean(x_spacing)
        x_std = np.std(x_spacing)
    else:
        x_period = 100  # Default value if we can't detect
        x_std = 0
    
    # For Y axis (logg)
    y_unique = np.unique(y_values)
    if len(y_unique) > 5:  # Need enough points for meaningful FFT
        y_spacing = np.diff(np.sort(y_unique))
        y_period = np.mean(y_spacing)
        y_std = np.std(y_spacing)
    else:
        y_period = 0.05  # Default value if we can't detect
        y_std = 0
    
    # Advanced periodicity detection using FFT for the z-values
    xi = np.linspace(min(x_values), max(x_values), 100)
    yi = np.linspace(min(y_values), max(y_values), 100)
    zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
    
    # Run FFT on both dimensions
    x_fft = np.abs(fft(np.nanmean(zi, axis=0)))
    # Add a check before calculating the mean:
    if np.any(~np.isnan(zi)):
        y_fft = np.abs(fft(np.nanmean(zi, axis=1)))
    else:
        # Handle the empty case
        y_fft = np.zeros(len(yi))
    
    x_freqs = fftfreq(len(xi), (max(xi) - min(xi)) / len(xi))
    y_freqs = fftfreq(len(yi), (max(yi) - min(yi)) / len(yi))
    
    # Find peaks in FFT to detect periodicities
    x_peaks = signal.find_peaks(x_fft)[0]
    y_peaks = signal.find_peaks(y_fft)[0]
    
    # Get the most significant peak
    x_periodicity = None
    y_periodicity = None
    
    if len(x_peaks) > 0 and len(x_freqs) > max(x_peaks):
        significant_x_peaks = x_peaks[1:len(x_peaks)//2]  # Skip DC component
        if len(significant_x_peaks) > 0:
            most_sig_x = significant_x_peaks[np.argmax(x_fft[significant_x_peaks])]
            if x_freqs[most_sig_x] != 0:
                x_periodicity = 1.0 / abs(x_freqs[most_sig_x])
    
    if len(y_peaks) > 0 and len(y_freqs) > max(y_peaks):
        significant_y_peaks = y_peaks[1:len(y_peaks)//2]  # Skip DC component
        if len(significant_y_peaks) > 0:
            most_sig_y = significant_y_peaks[np.argmax(y_fft[significant_y_peaks])]
            if y_freqs[most_sig_y] != 0:
                y_periodicity = 1.0 / abs(y_freqs[most_sig_y])
    
    # If FFT didn't work well, fall back to the simpler spacing method
    if x_periodicity is None or not np.isfinite(x_periodicity) or x_periodicity > (max(x_values) - min(x_values)):
        x_periodicity = x_period
    
    if y_periodicity is None or not np.isfinite(y_periodicity) or y_periodicity > (max(y_values) - min(y_values)):
        y_periodicity = y_period
    
    return x_periodicity, y_periodicity, x_std, y_std

def find_patterns_in_quality(x_values, y_values, z_values):
    """Find patterns or clusters in the quality values."""
    # Normalize the values for clustering
    data_for_clustering = np.column_stack([
        (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values)),
        (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values)),
        (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
    ])
    
    # Use DBSCAN to find clusters
    clustering = DBSCAN(eps=0.15, min_samples=3).fit(data_for_clustering)
    cluster_labels = clustering.labels_
    
    # Count valid clusters (excluding noise points labeled as -1)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    
    return cluster_labels, n_clusters

def ml_predict_points(x_values, y_values, z_values, x_range, y_range, n_points=200):
    """Use machine learning to predict promising points for additional data collection."""
    # Normalize the data
    x_norm = (x_values - np.min(x_values)) / (np.max(x_values) - np.min(x_values))
    y_norm = (y_values - np.min(y_values)) / (np.max(y_values) - np.min(y_values))
    
    # Create input data matrix
    X = np.column_stack([x_norm, y_norm])
    
    # Apply anomaly detection to find regions of interest
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    isolation_forest.fit(X)
    
    # Create a grid of potential points
    grid_size = 100
    x_grid = np.linspace(0, 1, grid_size)
    y_grid = np.linspace(0, 1, grid_size)
    XX, YY = np.meshgrid(x_grid, y_grid)
    grid_points = np.column_stack([XX.ravel(), YY.ravel()])
    
    # Predict anomaly scores
    anomaly_scores = isolation_forest.decision_function(grid_points)
    
    # Try to use Gaussian Process for uncertainty estimation
    try:
        # Define kernel
        kernel = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True, random_state=42)
        
        # Fit GP
        gp.fit(X, z_values)
        
        # Predict mean and std for grid points
        mean_pred, std_pred = gp.predict(grid_points, return_std=True)
        
        # Combine anomaly scores and uncertainty for interesting points
        combined_score = std_pred * (1 - (anomaly_scores - np.min(anomaly_scores)) / 
                                   (np.max(anomaly_scores) - np.min(anomaly_scores)))
    except:
        # Fallback if GP fails
        print("Warning: Gaussian Process failed, using anomaly scores only")
        combined_score = -anomaly_scores  # Negative because lower anomaly scores are more anomalous
    
    # Select top points based on combined score
    top_indices = np.argsort(combined_score)[-n_points:]
    top_points = grid_points[top_indices]
    
    # Scale back to original range
    pred_x = top_points[:, 0] * (np.max(x_values) - np.min(x_values)) + np.min(x_values)
    pred_y = top_points[:, 1] * (np.max(y_values) - np.min(y_values)) + np.min(y_values)
    
    # Adjust for specified ranges
    if x_range is not None:
        x_min, x_max = x_range
        pred_x = np.clip(pred_x, x_min, x_max)
    
    if y_range is not None:
        y_min, y_max = y_range
        pred_y = np.clip(pred_y, y_min, y_max)
    
    return pred_x, pred_y

def predict_additional_points(x_values, y_values, z_values, x_periodicity, y_periodicity, 
                             x_margin=200, y_margin=0.1, x_range=None, y_range=None):
    """Predict additional points based on detected periodicities."""
    # Define bounds for prediction
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    
    # If ranges are provided, use them instead
    if x_range is not None:
        x_ext_min, x_ext_max = x_range
    else:
        x_ext_min, x_ext_max = x_min - x_margin, x_max + x_margin
    
    if y_range is not None:
        y_ext_min, y_ext_max = y_range
    else:
        y_ext_min, y_ext_max = y_min - y_margin, y_max + y_margin
    
    # Prepare lists for predicted points
    pred_x = []
    pred_y = []
    
    # Generate predicted points beyond the current grid
    # X-direction (Teff)
    if x_periodicity and x_periodicity > 0:
        for x in np.arange(x_min - x_periodicity, x_ext_min, -x_periodicity):
            for y in np.linspace(y_min, y_max, 10):
                if y_ext_min <= y <= y_ext_max:
                    pred_x.append(x)
                    pred_y.append(y)
        
        for x in np.arange(x_max + x_periodicity, x_ext_max, x_periodicity):
            for y in np.linspace(y_min, y_max, 10):
                if y_ext_min <= y <= y_ext_max:
                    pred_x.append(x)
                    pred_y.append(y)
    
    # Y-direction (logg)
    if y_periodicity and y_periodicity > 0:
        for y in np.arange(y_min - y_periodicity, y_ext_min, -y_periodicity):
            for x in np.linspace(x_min, x_max, 10):
                if x_ext_min <= x <= x_ext_max:
                    pred_x.append(x)
                    pred_y.append(y)
        
        for y in np.arange(y_max + y_periodicity, y_ext_max, y_periodicity):
            for x in np.linspace(x_min, x_max, 10):
                if x_ext_min <= x <= x_ext_max:
                    pred_x.append(x)
                    pred_y.append(y)
    
    return np.array(pred_x), np.array(pred_y)

def plot_circular_wrap(x_values, y_values, z_values, pred_x, pred_y, x_periodicity, 
                      y_periodicity, cluster_labels, n_clusters, z_scale, output_file, 
                      x_range=None, y_range=None, plot_type='standard'):
    """Create a visualization with circular wrap to better show periodicities."""
    if plot_type == 'circular':
        plt.figure(figsize=(14, 14))
        ax1 = plt.subplot(111, projection='polar')  # Proper polar projection
        
        # Convert to polar coordinates
        # Normalize values first
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        
        # Apply ranges if specified
        if x_range:
            x_min, x_max = x_range
        if y_range:
            y_min, y_max = y_range
        
        # Check for division by zero and handle it
        x_range_size = x_max - x_min
        y_range_size = y_max - y_min
        
        if x_range_size <= 0:
            x_range_size = 1.0  # Avoid division by zero
            x_min -= 0.5
            x_max += 0.5
        
        if y_range_size <= 0:
            y_range_size = 1.0  # Avoid division by zero
            y_min -= 0.5
            y_max += 0.5
        
        # Normalize points to 0-1 range
        x_norm = (x_values - x_min) / x_range_size
        y_norm = (y_values - y_min) / y_range_size
        
        # Convert to polar coordinates (r, theta)
        # Use x for radius, y for angle to create a spiral
        r = 0.3 + 0.7 * x_norm
        theta = 2 * np.pi * y_norm
        
        # Create colormap
        colors = ['darkviolet', 'navy', 'teal', 'green', 'yellowgreen', 'yellow']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Plot original points in polar coordinates
        scatter = ax1.scatter(theta, r, c=z_values, cmap=cmap, s=100, marker='o', 
                              edgecolor='white', linewidth=1.5)
        plt.colorbar(scatter, label='Quality (Shifted)')
        
        # Plot predicted points if available
        if len(pred_x) > 0 and len(pred_y) > 0:
            # Handle division by zero in prediction data
            pred_x_norm = np.zeros_like(pred_x)
            pred_y_norm = np.zeros_like(pred_y)
            
            # Safe normalization
            mask_valid_x = np.isfinite(pred_x)
            mask_valid_y = np.isfinite(pred_y)
            mask_valid = mask_valid_x & mask_valid_y
            
            if np.any(mask_valid):
                pred_x_norm[mask_valid] = (pred_x[mask_valid] - x_min) / x_range_size
                pred_y_norm[mask_valid] = (pred_y[mask_valid] - y_min) / y_range_size
                
                # Ensure values are in 0-1 range
                pred_x_norm = np.clip(pred_x_norm, 0, 1)
                pred_y_norm = np.clip(pred_y_norm, 0, 1)
                
                pred_r = 0.3 + 0.7 * pred_x_norm
                pred_theta = 2 * np.pi * pred_y_norm
                
                ax1.scatter(pred_theta, pred_r, c='red', marker='x', s=40, alpha=0.7, 
                          label='Predicted Points')
        
        # Highlight clusters if they exist
        unique_labels = set(cluster_labels)
        if len(unique_labels) > 1:
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                mask = cluster_labels == label
                cluster_theta = theta[mask]
                cluster_r = r[mask]
                ax1.scatter(cluster_theta, cluster_r, marker='*', s=150,
                          edgecolor='white', linewidth=1.5, 
                          label=f'Cluster {label}' if label == min([l for l in unique_labels if l >= 0]) else None)
        
        # Add grid lines
        for i in np.linspace(0, 1, 6):
            circle_r = 0.3 + 0.7 * i
            circle = plt.Circle((0, 0), circle_r, fill=False, color='gray', linestyle='--', alpha=0.5)
            ax1.add_patch(circle)
        
        for i in np.linspace(0, 2*np.pi, 12, endpoint=False):
            ax1.plot([i, i], [0.3, 1.0], 'gray', linestyle='--', alpha=0.5)
        
        # Set plot properties - do NOT use plt.axis('equal') for polar plots
        ax1.set_theta_zero_location('N')
        ax1.set_title(f'Circular Periodicity Analysis (z_scale={z_scale})\nTeff={x_periodicity:.2f}, logg={y_periodicity:.4f}', fontsize=14)
        
        # Add text labels for Teff (radial) and logg (angular)
        for i, val in enumerate(np.linspace(x_min, x_max, 6)):
            r_pos = 0.3 + 0.7 * (i/5)
            ax1.text(0, r_pos, f'Teff: {val:.0f}', ha='center', va='center', 
                    bbox=dict(facecolor='white', alpha=0.7))
        
        for i, val in enumerate(np.linspace(y_min, y_max, 12, endpoint=False)):
            angle = 2 * np.pi * (i/12)
            ax1.text(angle, 1.1, f'logg: {val:.2f}', ha='center', va='center', 
                    rotation=np.degrees(angle), rotation_mode='anchor',
                    bbox=dict(facecolor='white', alpha=0.7))
        
        plt.legend(loc='lower right', bbox_to_anchor=(1.3, 0.0), fontsize=10)
        
    else:  # Standard plot
        plt.figure(figsize=(12, 10))
        
        # Create a custom colormap
        colors = ['darkviolet', 'navy', 'teal', 'green', 'yellowgreen', 'yellow']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Calculate bounds for the plot
        x_min, x_max = min(min(x_values), min(pred_x) if len(pred_x) > 0 else float('inf')), max(max(x_values), max(pred_x) if len(pred_x) > 0 else float('-inf'))
        y_min, y_max = min(min(y_values), min(pred_y) if len(pred_y) > 0 else float('inf')), max(max(y_values), max(pred_y) if len(pred_y) > 0 else float('-inf'))
        
        # Apply ranges if specified
        if x_range:
            x_min, x_max = x_range
        if y_range:
            y_min, y_max = y_range
        
        # Add some padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        if x_range > 0:
            x_min -= 0.05 * x_range
            x_max += 0.05 * x_range
        if y_range > 0:
            y_min -= 0.05 * y_range
            y_max += 0.05 * y_range
        
        # Create a grid for the contour plot
        xi = np.linspace(min(x_values), max(x_values), 100)
        yi = np.linspace(min(y_values), max(y_values), 100)
        zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
        
        # Plot the contour
        contour = plt.contourf(xi, yi, zi, 100, cmap=cmap)
        plt.colorbar(contour, label='Quality (Shifted)')
        
        # Plot the original data points
        plt.scatter(x_values, y_values, marker='o', s=60, 
                    edgecolor='white', facecolor='black', linewidth=1.5,
                    label='Original Data Points')
        
        # Highlight detected clusters if they exist
        unique_labels = set(cluster_labels)
        if len(unique_labels) > 1:  # If we have more than just noise (-1)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                mask = cluster_labels == label
                plt.scatter(x_values[mask], y_values[mask], marker='*', s=150,
                            edgecolor='white', linewidth=1.5, 
                            label=f'Cluster {label}' if label == min([l for l in unique_labels if l >= 0]) else None)
        
        # Improve visualization of predicted points - use density heatmap instead of scattered points
        if len(pred_x) > 0:
            # Create a 2D histogram of predicted points
            pred_hist, xedges, yedges = np.histogram2d(
                pred_x, pred_y, 
                bins=[np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)]
            )
            
            # Plot as a heatmap with transparency
            extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
            plt.imshow(pred_hist.T, extent=extent, origin='lower', 
                      cmap='Reds', alpha=0.6, aspect='auto')
            
            # Add a small sample of the actual predicted points for reference
            sample_size = min(50, len(pred_x))
            sample_indices = np.random.choice(len(pred_x), sample_size, replace=False)
            plt.scatter(pred_x[sample_indices], pred_y[sample_indices], 
                      c='red', marker='x', s=40, 
                      label='Predicted Points (Sample)')
            
            # Add contour lines to show prediction density
            pred_x_grid = np.linspace(x_min, x_max, 50)
            pred_y_grid = np.linspace(y_min, y_max, 50)
            pred_hist_smooth = np.zeros((50, 50))
            
            # Create a smoothed density map
            for i in range(len(pred_x)):
                x_idx = np.argmin(np.abs(pred_x_grid - pred_x[i]))
                y_idx = np.argmin(np.abs(pred_y_grid - pred_y[i]))
                if 0 <= x_idx < 50 and 0 <= y_idx < 50:
                    pred_hist_smooth[x_idx, y_idx] += 1
            
            # Apply Gaussian smoothing
            pred_hist_smooth = ndimage.gaussian_filter(pred_hist_smooth, sigma=1.0)
            
            # Plot contour lines
            X_grid, Y_grid = np.meshgrid(pred_x_grid, pred_y_grid)
            plt.contour(X_grid, Y_grid, pred_hist_smooth.T, 
                       levels=5, colors='red', alpha=0.7, 
                       linestyles='dashed', linewidths=1)
        
        # Add grid lines
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Set labels and title
        plt.xlabel('Effective Temperature (Teff)', fontsize=12)
        plt.ylabel('Surface Gravity (log g)', fontsize=12)
        plt.title(f'Periodicity Analysis for Stellar Atmosphere Models (z_scale={z_scale})\nDetected Periodicities: Teff={x_periodicity:.2f}, logg={y_periodicity:.4f}', fontsize=14)
        
        # Add text box with periodicity info
        info_str = f'Detected Teff Periodicity: {x_periodicity:.2f}\n'
        info_str += f'Detected logg Periodicity: {y_periodicity:.4f}\n'
        info_str += f'Number of Pattern Clusters: {n_clusters}\n'
        info_str += f'Suggested Points: {len(pred_x)}'
        
        plt.annotate(info_str, xy=(0.02, 0.02), xycoords='axes fraction',
                     fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add legend
        plt.legend(loc='upper right', fontsize=10)
        
        # Set axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
    # Adjust the plot layout
    plt.tight_layout()
    
    # Save to file
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    return

def main():
    parser = argparse.ArgumentParser(description='Analyze and visualize periodicities in model data.')
    parser.add_argument('json_file', help='Input JSON file with model data')
    parser.add_argument('--output', '-o', help='Output PNG file', default="periodicities.png")
    parser.add_argument('--teff_min', type=float, help='Minimum Teff value for plotting', default=7000)
    parser.add_argument('--teff_max', type=float, help='Maximum Teff value for plotting', default=9000)
    parser.add_argument('--logg_min', type=float, help='Minimum logg value for plotting', default=2)
    parser.add_argument('--logg_max', type=float, help='Maximum logg value for plotting', default=6)
    parser.add_argument('--plot_type', choices=['standard', 'circular'], default='standard', 
                      help='Type of plot visualization')
    parser.add_argument('--use_ml', action='store_true', help='Use machine learning for predictions')
    args = parser.parse_args()
    
    # Set output file name based on input if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        args.output = f"{base_name}_periodicities_{args.plot_type}.png"
    
    # Set range variables
    x_range = None
    if args.teff_min is not None and args.teff_max is not None:
        x_range = (args.teff_min, args.teff_max)
    
    y_range = None
    if args.logg_min is not None and args.logg_max is not None:
        y_range = (args.logg_min, args.logg_max)
    
    # Load data
    data = load_data(args.json_file)
    
    # Extract data
    teff_values, logg_values, quality_shifted_values, z_scale_values = extract_grid_data(data)
    
    # Get most common z_scale for the title
    z_scale = z_scale_values[0] if len(z_scale_values) > 0 else "N/A"
    
    # Detect periodicities
    x_periodicity, y_periodicity, x_std, y_std = detect_periodicities(
        teff_values, logg_values, quality_shifted_values)
    
    # Find patterns in quality values
    cluster_labels, n_clusters = find_patterns_in_quality(
        teff_values, logg_values, quality_shifted_values)
    
    # Choose prediction method
    if args.use_ml:
        print("Using machine learning for point prediction...")
        pred_x, pred_y = ml_predict_points(
            teff_values, logg_values, quality_shifted_values, 
            x_range, y_range, n_points=200)
    else:
        # Predict additional points based on periodicity
        pred_x, pred_y = predict_additional_points(
            teff_values, logg_values, quality_shifted_values, 
            x_periodicity, y_periodicity, x_range=x_range, y_range=y_range)
    
    # Create the plot
    plot_circular_wrap(
        teff_values, logg_values, quality_shifted_values,
        pred_x, pred_y, x_periodicity, y_periodicity,
        cluster_labels, n_clusters, z_scale, args.output,
        x_range=x_range, y_range=y_range, plot_type=args.plot_type)
    
    # Print detected periodicities
    print(f"Detected periodicities:")
    print(f"  Teff periodicity: {x_periodicity:.2f} ± {x_std:.2f}")
    print(f"  logg periodicity: {y_periodicity:.4f} ± {y_std:.4f}")
    print(f"  Number of pattern clusters: {n_clusters}")
    print(f"Suggested additional data points: {len(pred_x)}")
    
    # Print some recommendations for further data collection
    print("\nRecommendations for further data collection:")
    
    # Group predictions by Teff
    teff_groups = {}
    for i in range(len(pred_x)):
        teff = pred_x[i]
        logg = pred_y[i]
        if teff not in teff_groups:
            teff_groups[teff] = []
        teff_groups[teff].append(logg)
    
    # Print the top 5 most promising Teff values with their logg ranges
    print("Most promising regions to explore:")
    for i, (teff, loggs) in enumerate(sorted(teff_groups.items())[:5]):
        logg_min, logg_max = min(loggs), max(loggs)
        print(f"  Region {i+1}: Teff = {teff:.1f}, logg range: {logg_min:.2f}-{logg_max:.2f}")
    
    # Export predicted points to a CSV file if there are any
    if len(pred_x) > 0:
        csv_filename = os.path.splitext(args.output)[0] + "_predicted_points.csv"
        with open(csv_filename, 'w') as f:
            f.write("Teff,logg\n")
            for i in range(len(pred_x)):
                f.write(f"{pred_x[i]:.2f},{pred_y[i]:.4f}\n")
        print(f"Predicted points exported to {csv_filename}")
    
    # Add a summary recommendation
    if n_clusters > 0:
        print("\nThe data shows distinct cluster patterns. We recommend focused sampling in these clustered regions.")
        print(f"Use the predicted points in the exported CSV file for guidance.")
        print(f"For best results, collect at least {max(10, len(pred_x) // 10)} additional data points.")
    else:
        print("\nNo strong clustering detected. We recommend sampling according to the detected periodicities:")
        print(f"  - Sample at Teff intervals of approximately {x_periodicity:.1f}")
        print(f"  - Sample at logg intervals of approximately {y_periodicity:.4f}")
        print(f"  - Focus on sampling in the regions with predicted points (see CSV file)")

if __name__ == "__main__":
    main()