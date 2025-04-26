#!/usr/bin/env python3
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import signal
from scipy.interpolate import griddata
import pywt  # For wavelet analysis
import argparse
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import HDBSCAN  # Hierarchical DBSCAN for better clustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel, Matern, ExpSineSquared
)
import os
import matplotlib as mpl
from sklearn.manifold import TSNE  # For dimensionality reduction visualization
from tqdm import tqdm  # For progress bars

# Set high-quality plot defaults
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 12

class Autoencoder(nn.Module):
    """
    Neural network autoencoder for anomaly detection and feature learning.
    The network compresses input data to a lower-dimensional latent space,
    then reconstructs it. Points with high reconstruction error are anomalies.
    """
    def __init__(self, input_dim, encoding_dim=8):
        super(Autoencoder, self).__init__()
        # Encoder with decreasing layer sizes
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        # Decoder with increasing layer sizes
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


def train_autoencoder(x_values, y_values, z_values, encoding_dim=8, epochs=300):
    """
    Train an autoencoder for anomaly detection and feature learning.
    
    Returns:
        model: Trained autoencoder model
        scaler: Fitted data scaler
        reconstruction_errors: Error for each input data point
    """
    # Normalize the data
    scaler = MinMaxScaler()
    data = scaler.fit_transform(np.column_stack([x_values, y_values, z_values]))
    
    # Convert to PyTorch tensors
    tensor_data = torch.FloatTensor(data)
    dataset = TensorDataset(tensor_data, tensor_data)
    dataloader = DataLoader(dataset, batch_size=min(32, len(data)), shuffle=True)
    
    # Initialize model
    input_dim = data.shape[1]
    model = Autoencoder(input_dim, encoding_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with progress tracking
    print("Training autoencoder model...")
    model.train()
    for epoch in tqdm(range(epochs), desc="Training epochs"):
        total_loss = 0
        for batch_features, _ in dataloader:
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_features)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Print progress occasionally
        if (epoch+1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.6f}")
    
    # Calculate reconstruction error for each data point
    model.eval()
    with torch.no_grad():
        reconstructions = model(tensor_data)
        reconstruction_errors = torch.mean((reconstructions - tensor_data)**2, dim=1).numpy()
    
    return model, scaler, reconstruction_errors


def detect_periodicities_wavelet(x_values, y_values, z_values):
    """
    Detect periodicities using wavelet analysis, which is more robust for 
    irregular data than traditional FFT.
    
    Returns:
        x_periodicity: Detected periodicity in x dimension
        y_periodicity: Detected periodicity in y dimension
        x_std: Standard deviation of x periodicity
        y_std: Standard deviation of y periodicity
    """
    print("Detecting periodicities with wavelet analysis...")
    
    # Simple spacing method as fallback
    x_unique = np.sort(np.unique(x_values))
    y_unique = np.sort(np.unique(y_values))
    
    x_spacing = np.diff(x_unique) if len(x_unique) > 1 else [100]
    y_spacing = np.diff(y_unique) if len(y_unique) > 1 else [0.05]
    
    # Default values in case advanced methods fail
    x_period_default = np.mean(x_spacing) if len(x_spacing) > 0 else 100
    y_period_default = np.mean(y_spacing) if len(y_spacing) > 0 else 0.05
    x_std_default = np.std(x_spacing) if len(x_spacing) > 1 else 0
    y_std_default = np.std(y_spacing) if len(y_spacing) > 1 else 0
    
    # Check if we have enough unique points
    if len(x_unique) < 5 or len(y_unique) < 5:
        print("Warning: Not enough unique points for wavelet analysis")
        return x_period_default, y_period_default, x_std_default, y_std_default
    
    try:
        # Interpolate to regular grid for analysis
        x_grid_size = min(128, max(64, 2 ** int(np.log2(len(x_unique)) + 1)))
        y_grid_size = min(128, max(64, 2 ** int(np.log2(len(y_unique)) + 1)))
        
        xi = np.linspace(min(x_values), max(x_values), x_grid_size)
        yi = np.linspace(min(y_values), max(y_values), y_grid_size)
        
        # Interpolate z values onto a regular grid
        zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic', fill_value=np.nan)
        
        # Replace NaN values with mean of neighbors
        mask = np.isnan(zi)
        if np.all(mask):
            return x_period_default, y_period_default, x_std_default, y_std_default
        
        zi[mask] = np.nanmean(zi)
        
        # Store results from wavelet analysis
        x_periods = []
        x_powers = []
        y_periods = []
        y_powers = []
        
        # Analyze rows (Teff dimension)
        for i in range(zi.shape[0]):
            row = zi[i, :]
            if np.all(np.isnan(row)):
                continue
                
            # Apply Continuous Wavelet Transform
            try:
                scales = np.arange(1, min(64, len(row)//2))
                wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
                coeffs, freqs = pywt.cwt(row, scales, wavelet)
                
                # Calculate power
                power = np.abs(coeffs)**2
                # Find the scale with maximum power
                scale_idx = np.unravel_index(np.argmax(power), power.shape)[0]
                # Convert scale to period in terms of data spacing
                dx = (max(xi) - min(xi)) / (len(xi) - 1)
                period = scales[scale_idx] * dx
                
                # Only add periods that are reasonable
                if period < (max(xi) - min(xi)) / 2 and period > dx * 2:
                    x_periods.append(period)
                    x_powers.append(np.max(power))
            except Exception as e:
                print(f"Warning: Wavelet analysis failed for row {i}: {e}")
        
        # Analyze columns (logg dimension)
        for j in range(zi.shape[1]):
            col = zi[:, j]
            if np.all(np.isnan(col)):
                continue
                
            # Apply Continuous Wavelet Transform
            try:
                scales = np.arange(1, min(64, len(col)//2))
                wavelet = 'cmor1.5-1.0'  # Complex Morlet wavelet
                coeffs, freqs = pywt.cwt(col, scales, wavelet)
                
                # Calculate power
                power = np.abs(coeffs)**2
                # Find the scale with maximum power
                scale_idx = np.unravel_index(np.argmax(power), power.shape)[0]
                # Convert scale to period in terms of data spacing
                dy = (max(yi) - min(yi)) / (len(yi) - 1)
                period = scales[scale_idx] * dy
                
                # Only add periods that are reasonable
                if period < (max(yi) - min(yi)) / 2 and period > dy * 2:
                    y_periods.append(period)
                    y_powers.append(np.max(power))
            except Exception as e:
                print(f"Warning: Wavelet analysis failed for column {j}: {e}")
        
        # Also try FFT for comparison
        # Calculate mean along rows (for x/Teff periodicity)
        x_mean = np.nanmean(zi, axis=0)
        x_fft = np.abs(np.fft.rfft(x_mean - np.nanmean(x_mean)))
        x_freqs = np.fft.rfftfreq(len(x_mean), d=(max(xi) - min(xi)) / (len(xi) - 1))
        
        # Calculate mean along columns (for y/logg periodicity)
        y_mean = np.nanmean(zi, axis=1)
        y_fft = np.abs(np.fft.rfft(y_mean - np.nanmean(y_mean)))
        y_freqs = np.fft.rfftfreq(len(y_mean), d=(max(yi) - min(yi)) / (len(yi) - 1))
        
        # Find peaks in FFT power spectrum
        x_peaks = signal.find_peaks(x_fft, height=np.max(x_fft)/10)[0]
        y_peaks = signal.find_peaks(y_fft, height=np.max(y_fft)/10)[0]
        
        # Calculate FFT-based periods
        x_fft_periods = []
        x_fft_powers = []
        for peak in x_peaks:
            if peak > 0 and peak < len(x_freqs):  # Skip DC component
                period = 1.0 / x_freqs[peak]
                if period < (max(xi) - min(xi)) / 2:  # Avoid periods larger than half the domain
                    x_fft_periods.append(period)
                    x_fft_powers.append(x_fft[peak])
        
        y_fft_periods = []
        y_fft_powers = []
        for peak in y_peaks:
            if peak > 0 and peak < len(y_freqs):  # Skip DC component
                period = 1.0 / y_freqs[peak]
                if period < (max(yi) - min(yi)) / 2:  # Avoid periods larger than half the domain
                    y_fft_periods.append(period)
                    y_fft_powers.append(y_fft[peak])
        
        # Combine results from wavelet and FFT analyses
        combined_x_periods = x_periods + x_fft_periods
        combined_x_powers = x_powers + x_fft_powers
        combined_y_periods = y_periods + y_fft_periods
        combined_y_powers = y_powers + y_fft_powers
        
        # Calculate weighted average of periods based on power
        if combined_x_periods:
            x_periodicity = np.average(combined_x_periods, weights=combined_x_powers)
            x_std = np.sqrt(np.average((combined_x_periods - x_periodicity)**2, weights=combined_x_powers))
        else:
            x_periodicity = x_period_default
            x_std = x_std_default
            
        if combined_y_periods:
            y_periodicity = np.average(combined_y_periods, weights=combined_y_powers)
            y_std = np.sqrt(np.average((combined_y_periods - y_periodicity)**2, weights=combined_y_powers))
        else:
            y_periodicity = y_period_default
            y_std = y_std_default
        
        # Validate results
        if not np.isfinite(x_periodicity) or x_periodicity <= 0 or x_periodicity > (max(x_values) - min(x_values)):
            print("Warning: Invalid x periodicity detected, using default")
            x_periodicity = x_period_default
            x_std = x_std_default
            
        if not np.isfinite(y_periodicity) or y_periodicity <= 0 or y_periodicity > (max(y_values) - min(y_values)):
            print("Warning: Invalid y periodicity detected, using default")
            y_periodicity = y_period_default
            y_std = y_std_default
        
        print(f"Detected x periodicity: {x_periodicity:.2f} ± {x_std:.2f}")
        print(f"Detected y periodicity: {y_periodicity:.4f} ± {y_std:.4f}")
        
        return x_periodicity, y_periodicity, x_std, y_std
        
    except Exception as e:
        print(f"Warning: Advanced periodicity detection failed: {e}")
        return x_period_default, y_period_default, x_std_default, y_std_default

def advanced_clustering(x_values, y_values, z_values, reconstruction_errors=None):
    """
    Perform advanced clustering using HDBSCAN, which is more robust than DBSCAN
    for datasets with varying densities.
    
    Parameters:
        x_values, y_values, z_values: Input data
        reconstruction_errors: Optional errors from autoencoder to enhance clustering
        
    Returns:
        cluster_labels: Cluster assignment for each data point
        n_clusters: Number of distinct clusters found
    """
    print("Performing advanced clustering...")
    
    # Normalize data for clustering
    scaler = StandardScaler()
    
    if reconstruction_errors is not None:
        # Include reconstruction error as a feature for clustering
        data_for_clustering = np.column_stack([
            x_values, y_values, z_values, reconstruction_errors
        ])
    else:
        data_for_clustering = np.column_stack([x_values, y_values, z_values])
    
    # Scale features
    data_scaled = scaler.fit_transform(data_for_clustering)
    
    # Apply HDBSCAN clustering
    clusterer = HDBSCAN(
        min_cluster_size=max(3, len(x_values) // 20),  # Adaptive min cluster size
        min_samples=max(2, len(x_values) // 30),       # Adaptive min samples
        cluster_selection_epsilon=0.5,                # Allow some noise tolerance
    )
    
    try:
        cluster_labels = clusterer.fit_predict(data_scaled)
        
        # Count valid clusters (excluding noise points labeled as -1)
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        
        print(f"Found {n_clusters} clusters")
        if n_clusters > 0:
            for i in range(n_clusters):
                cluster_size = np.sum(cluster_labels == i)
                print(f"  Cluster {i}: {cluster_size} points")
        
        return cluster_labels, n_clusters
    
    except Exception as e:
        print(f"Warning: Clustering failed: {e}")
        # Fallback to simple clustering
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=3).fit(data_scaled)
        cluster_labels = clustering.labels_
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        return cluster_labels, n_clusters

def create_gaussian_process_model(x_values, y_values, z_values, x_periodicity, y_periodicity):
    """
    Create a Gaussian Process model with a specialized kernel
    that incorporates detected periodicities.
    
    Parameters:
        x_values, y_values, z_values: Input data points
        x_periodicity, y_periodicity: Detected periodicities to incorporate into kernel
    
    Returns:
        gp: Trained Gaussian Process model
        x_scaler, y_scaler: Scalers for input normalization
    """
    print("Training Gaussian Process model...")
    
    # Normalize input data
    x_scaler = MinMaxScaler()
    y_scaler = MinMaxScaler()
    
    X = np.column_stack([x_values, y_values])
    X_scaled = np.column_stack([
        x_scaler.fit_transform(x_values.reshape(-1, 1)).flatten(),
        y_scaler.fit_transform(y_values.reshape(-1, 1)).flatten()
    ])
    
    # Define a kernel that can capture both periodicities and local variations
    # ExpSineSquared for periodicity, Matern for smoothness, WhiteKernel for noise
    k1 = ConstantKernel(1.0) * ExpSineSquared(length_scale=0.5, periodicity=0.1) * RBF(length_scale=0.5)
    k2 = ConstantKernel(1.0) * Matern(length_scale=0.5, nu=1.5) 
    k3 = WhiteKernel(noise_level=0.1)
    kernel = k1 + k2 + k3
    
    # Create and train the GP model
    gp = GaussianProcessRegressor(
        kernel=kernel, 
        alpha=1e-6,
        n_restarts_optimizer=5,
        normalize_y=True,
        random_state=42
    )
    
    try:
        gp.fit(X_scaled, z_values)
        print(f"Trained GP model with kernel: {gp.kernel_}")
        return gp, x_scaler, y_scaler
    
    except Exception as e:
        print(f"Warning: GP model training failed: {e}")
        # Fallback to simpler kernel
        simple_kernel = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=simple_kernel, alpha=1e-6, normalize_y=True)
        gp.fit(X_scaled, z_values)
        return gp, x_scaler, y_scaler

def neural_network_prediction(model, scaler, x_values, y_values, z_values, 
                             x_periodicity, y_periodicity, n_points=300, 
                             x_range=None, y_range=None):
    """
    Generate predictions for new points using neural network and periodicity information.
    
    Parameters:
        model: Trained autoencoder model
        scaler: Data scaler from model training
        x_values, y_values, z_values: Original data
        x_periodicity, y_periodicity: Detected periodicities
        n_points: Number of points to predict
        x_range, y_range: Range constraints
    
    Returns:
        pred_x, pred_y: Arrays of predicted point coordinates
    """
    print(f"Generating {n_points} predictions with neural network...")
    
    # Set exploration boundaries
    if x_range is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = min(x_values) - 1.5*x_periodicity, max(x_values) + 1.5*x_periodicity
    
    if y_range is not None:
        y_min, y_max = y_range
    else:
        y_min, y_max = min(y_values) - 1.5*y_periodicity, max(y_values) + 1.5*y_periodicity
    
    # Generate candidate points in a grid with periodicity-based spacing
    grid_size_x = max(20, int(np.ceil((x_max - x_min) / (x_periodicity / 4))))
    grid_size_y = max(20, int(np.ceil((y_max - y_min) / (y_periodicity / 4))))
    
    # Create grid of candidate points
    grid_x = np.linspace(x_min, x_max, grid_size_x)
    grid_y = np.linspace(y_min, y_max, grid_size_y)
    XX, YY = np.meshgrid(grid_x, grid_y)
    candidates_x = XX.flatten()
    candidates_y = YY.flatten()
    
    # Filter out points too close to existing data
    min_distances = []
    for i in range(len(candidates_x)):
        # Calculate distances to all existing points
        distances = np.sqrt((candidates_x[i] - x_values)**2 + 
                           (candidates_y[i] - y_values)**2)
        min_distances.append(np.min(distances))
    
    min_distances = np.array(min_distances)
    
    # Keep points with sufficient distance from existing data
    # but not too far (we want to explore the boundary)
    distance_threshold_min = min(x_periodicity/4, y_periodicity/2)
    distance_threshold_max = max(x_periodicity*2, y_periodicity*10)
    
    valid_mask = (min_distances > distance_threshold_min) & (min_distances < distance_threshold_max)
    filtered_x = candidates_x[valid_mask]
    filtered_y = candidates_y[valid_mask]
    
    # If no points passed the filter, use the original candidates
    if len(filtered_x) < 10:
        print("Warning: Few valid candidates after filtering, using full grid")
        filtered_x = candidates_x
        filtered_y = candidates_y
    
    # Use the autoencoder to evaluate candidates
    
    model.eval()
    try:
        interestingness_scores = []
        batch_size = 1000  # Process in batches to avoid memory issues
        
        for i in range(0, len(filtered_x), batch_size):
            # Prepare batch
            batch_end = min(i + batch_size, len(filtered_x))
            batch_x = filtered_x[i:batch_end]
            batch_y = filtered_y[i:batch_end]
            
            # Create dummy z-values (we'll use the average)
            batch_z = np.ones_like(batch_x) * np.mean(z_values)
            
            # Normalize data
            batch_data = np.column_stack([batch_x, batch_y, batch_z])
            batch_data_scaled = scaler.transform(batch_data)
            batch_tensor = torch.FloatTensor(batch_data_scaled)
            
            # Get encoded representation
            with torch.no_grad():
                batch_encoded = model.encode(batch_tensor).numpy()
                
                # Calculate distance to all training points in latent space
                # Points with intermediate distances are most interesting
                scores = []
                # Get encoded training data
                training_data = np.column_stack([x_values, y_values, z_values])
                training_data_scaled = scaler.transform(training_data)
                training_tensor = torch.FloatTensor(training_data_scaled)
                
                with torch.no_grad():
                    training_encoded = model.encode(training_tensor).numpy()
                
                # For each candidate, calculate minimum distance to training data in latent space
                for encoded_point in batch_encoded:
                    distances = np.sqrt(np.sum((training_encoded - encoded_point)**2, axis=1))
                    min_dist = np.min(distances)
                    # We want points that are not too close but not too far
                    # from the training data in the latent space
                    score = np.exp(-(min_dist - 1.0)**2 / 0.5)
                    scores.append(score)
                
                interestingness_scores.extend(scores)
        
        interestingness_scores = np.array(interestingness_scores)
        
        # Select top points based on scores
        n_select = min(n_points, len(filtered_x))
        top_indices = np.argsort(interestingness_scores)[-n_select:]
        pred_x = filtered_x[top_indices]
        pred_y = filtered_y[top_indices]
        
        return pred_x, pred_y

        
    except Exception as e:
        print(f"Neural network prediction failed: {e}")
        # Fallback to simple prediction
        n_select = min(n_points, len(filtered_x))
        selected_indices = np.random.choice(len(filtered_x), n_select, replace=False)
        pred_x = filtered_x[selected_indices]
        pred_y = filtered_y[selected_indices]
        return pred_x, pred_y

def plot_analysis_results(x_values, y_values, z_values, pred_x, pred_y, 
                         x_periodicity, y_periodicity, cluster_labels, n_clusters,
                         z_scale, output_file, x_range=None, y_range=None, 
                         reconstruction_errors=None):
    """
    Create an advanced visualization of the analysis results with multiple subplots.
    
    Parameters:
        x_values, y_values, z_values: Original data points
        pred_x, pred_y: Predicted new points
        x_periodicity, y_periodicity: Detected periodicities
        cluster_labels: Cluster assignments
        n_clusters: Number of clusters
        z_scale: Scale factor from the data
        output_file: Path to save the plot
        x_range, y_range: Range for plotting
        reconstruction_errors: Optional errors from autoencoder for visualization
    """
    print(f"Creating visualization and saving to {output_file}...")
    
    # Set up figure with subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid for subplots: 2x2 grid
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])  # Main heatmap
    ax2 = fig.add_subplot(gs[0, 1])  # Periodicity visualization
    ax3 = fig.add_subplot(gs[1, 0])  # Predictions
    ax4 = fig.add_subplot(gs[1, 1])  # t-SNE visualization or cluster visualization
    
    # Define custom colormap for main plot
    colors = ['darkviolet', 'navy', 'teal', 'green', 'yellowgreen', 'yellow']
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
    
    # Plot 1: Main heatmap with data points
    xi = np.linspace(min(x_values), max(x_values), 200)
    yi = np.linspace(min(y_values), max(y_values), 200)
    zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
    
    im = ax1.contourf(xi, yi, zi, 100, cmap=cmap)
    fig.colorbar(im, ax=ax1, label='Quality (Shifted)')
    
    # Plot original data points with cluster colors
    if n_clusters > 0:
        # Color by cluster
        for label in range(n_clusters):
            mask = cluster_labels == label
            ax1.scatter(x_values[mask], y_values[mask], 
                      marker='o', s=60, label=f'Cluster {label}',
                      edgecolor='white', linewidth=1)
        
        # Plot noise points
        noise_mask = cluster_labels == -1
        if np.any(noise_mask):
            ax1.scatter(x_values[noise_mask], y_values[noise_mask], 
                      marker='o', s=40, color='gray', alpha=0.6,
                      edgecolor='white', linewidth=0.5, label='Noise')
    else:
        # If no clusters, just plot all points
        ax1.scatter(x_values, y_values, marker='o', s=60, 
                  edgecolor='white', color='black', linewidth=1,
                  label='Original Data')
    
    ax1.set_xlabel('Effective Temperature (Teff)', fontsize=12)
    ax1.set_ylabel('Surface Gravity (log g)', fontsize=12)
    ax1.set_title('Data Heatmap with Clusters', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(loc='best', fontsize=10)
    
    # Plot 2: Periodicity visualization
    # Create a grid showing the periodicity pattern
    if x_periodicity > 0 and y_periodicity > 0:
        # Generate grid with periodicities
        x_grid_size = 200
        y_grid_size = 200
        
        # Center of the data
        x_center = (min(x_values) + max(x_values)) / 2
        y_center = (min(y_values) + max(y_values)) / 2
        
        # Create grid
        x_grid = np.linspace(x_center - 3*x_periodicity, x_center + 3*x_periodicity, x_grid_size)
        y_grid = np.linspace(y_center - 3*y_periodicity, y_center + 3*y_periodicity, y_grid_size)
        XX, YY = np.meshgrid(x_grid, y_grid)
        
        # Create a periodic pattern
        ZZ = np.sin(2 * np.pi * XX / x_periodicity) * np.cos(2 * np.pi * YY / y_periodicity)
        
        # Plot
        im2 = ax2.imshow(ZZ, extent=[x_grid[0], x_grid[-1], y_grid[0], y_grid[-1]], 
                      aspect='auto', origin='lower', cmap='RdBu_r')
        fig.colorbar(im2, ax=ax2, label='Periodic Pattern')
        
        # Add lines showing periodicity distances
        # Horizontal lines (y periodicity)
        for i in range(-3, 4):
            y_line = y_center + i * y_periodicity
            if min(y_values) <= y_line <= max(y_values):
                ax2.axhline(y_line, color='black', linestyle='--', alpha=0.7)
        
        # Vertical lines (x periodicity)
        for i in range(-3, 4):
            x_line = x_center + i * x_periodicity
            if min(x_values) <= x_line <= max(x_values):
                ax2.axvline(x_line, color='black', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Effective Temperature (Teff)', fontsize=12)
    ax2.set_ylabel('Surface Gravity (log g)', fontsize=12)
    ax2.set_title(f'Detected Periodicities: Teff={x_periodicity:.2f}, logg={y_periodicity:.4f}', fontsize=14)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Predicted points visualization
    if len(pred_x) > 0:
        # Create density heatmap for predicted points
        pred_hist, xedges, yedges = np.histogram2d(
            pred_x, pred_y, 
            bins=[np.linspace(min(min(x_values), min(pred_x)), max(max(x_values), max(pred_x)), 50),
                 np.linspace(min(min(y_values), min(pred_y)), max(max(y_values), max(pred_y)), 50)]
        )
        
        # Plot as a heatmap
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        ax3.imshow(pred_hist.T, extent=extent, origin='lower', 
                 cmap='Reds', alpha=0.6, aspect='auto')
        
        # Plot a sample of the predicted points
        sample_size = min(50, len(pred_x))
        sample_indices = np.random.choice(len(pred_x), sample_size, replace=False)
        ax3.scatter(pred_x[sample_indices], pred_y[sample_indices], 
                  c='red', marker='x', s=40, label='Predicted Points (Sample)')
        
        # Also show original data points for reference
        ax3.scatter(x_values, y_values, marker='o', s=40, alpha=0.5,
                  edgecolor='white', color='black', linewidth=0.5,
                  label='Original Data')
    
    ax3.set_xlabel('Effective Temperature (Teff)', fontsize=12)
    ax3.set_ylabel('Surface Gravity (log g)', fontsize=12)
    ax3.set_title(f'Suggested Additional Data Points (n={len(pred_x)})', fontsize=14)
    ax3.grid(True, linestyle='--', alpha=0.7)
    ax3.legend(loc='best', fontsize=10)
    
    # Plot 4: Advanced visualization - t-SNE or cluster-focused
    if reconstruction_errors is not None and len(x_values) > 10:
        # Use t-SNE for dimensionality reduction visualization
        try:
            # Prepare data with more features
            tsne_data = np.column_stack([
                x_values, y_values, z_values, reconstruction_errors
            ])
            tsne_data_scaled = StandardScaler().fit_transform(tsne_data)
            
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=min(30, len(x_values)-1), 
                        random_state=42, n_iter=1000)
            tsne_result = tsne.fit_transform(tsne_data_scaled)
            
            # Plot t-SNE result
            if n_clusters > 0:
                # Color by cluster
                for label in range(n_clusters):
                    mask = cluster_labels == label
                    ax4.scatter(tsne_result[mask, 0], tsne_result[mask, 1], 
                              marker='o', s=60, label=f'Cluster {label}')
                
                # Plot noise points
                noise_mask = cluster_labels == -1
                if np.any(noise_mask):
                    ax4.scatter(tsne_result[noise_mask, 0], tsne_result[noise_mask, 1], 
                              marker='o', s=40, color='gray', alpha=0.6, label='Noise')
            else:
                # If no clusters, color by quality value
                scatter = ax4.scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                   c=z_values, cmap=cmap, s=60, 
                                   edgecolor='white', linewidth=1)
                fig.colorbar(scatter, ax=ax4, label='Quality (Shifted)')
            
            ax4.set_xlabel('t-SNE Dimension 1', fontsize=12)
            ax4.set_ylabel('t-SNE Dimension 2', fontsize=12)
            ax4.set_title('t-SNE Visualization of Data Structure', fontsize=14)
            if n_clusters > 0:
                ax4.legend(loc='best', fontsize=10)
            
        except Exception as e:
            print(f"Warning: t-SNE visualization failed: {e}")
            # Fallback to error plot
            if reconstruction_errors is not None:
                scatter = ax4.scatter(x_values, y_values, c=reconstruction_errors, 
                                   cmap='plasma', s=60, marker='o',
                                   edgecolor='white', linewidth=1)
                fig.colorbar(scatter, ax=ax4, label='Reconstruction Error')
                ax4.set_xlabel('Effective Temperature (Teff)', fontsize=12)
                ax4.set_ylabel('Surface Gravity (log g)', fontsize=12)
                ax4.set_title('Reconstruction Error Distribution', fontsize=14)
            else:
                # Just repeat the heatmap but with different color focus
                quality_norm = (z_values - np.min(z_values)) / (np.max(z_values) - np.min(z_values))
                scatter = ax4.scatter(x_values, y_values, c=quality_norm, 
                                   cmap='viridis', s=60, marker='o',
                                   edgecolor='white', linewidth=1)
                fig.colorbar(scatter, ax=ax4, label='Normalized Quality')
                ax4.set_xlabel('Effective Temperature (Teff)', fontsize=12)
                ax4.set_ylabel('Surface Gravity (log g)', fontsize=12)
                ax4.set_title('Alternative Quality Visualization', fontsize=14)
    else:
        # Fallback - just show a zoomed view of the main heatmap
        # but with focus on error regions
        if reconstruction_errors is not None:
            scatter = ax4.scatter(x_values, y_values, c=reconstruction_errors, 
                               cmap='plasma', s=60, marker='o',
                               edgecolor='white', linewidth=1)
            fig.colorbar(scatter, ax=ax4, label='Reconstruction Error')
        else:
            # Calculate a measure of "unexpectedness" - deviation from local average
            grid_z = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='linear')
            grid_z_smooth = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
            
            # Calculate the difference between linear and cubic interpolation
            # This highlights areas where the function is not well-behaved
            with np.errstate(invalid='ignore'):  # Ignore NaN warnings
                unexpectedness = np.abs(grid_z - grid_z_smooth)
                
            im4 = ax4.contourf(xi, yi, unexpectedness, 100, cmap='plasma')
            fig.colorbar(im4, ax=ax4, label='Interpolation Deviation')
        
        ax4.set_xlabel('Effective Temperature (Teff)', fontsize=12)
        ax4.set_ylabel('Surface Gravity (log g)', fontsize=12)
        ax4.set_title('Quality Uncertainty/Deviation', fontsize=14)
    
    ax4.grid(True, linestyle='--', alpha=0.7)
    
    # Add overall title
    fig.suptitle(f'Advanced Periodicity Analysis (z_scale={z_scale})', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])  # Make room for the suptitle
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    
    return

def gp_uncertainty_prediction(gp, x_scaler, y_scaler, x_values, y_values, z_values,
                            x_periodicity, y_periodicity, n_points=200, 
                            x_range=None, y_range=None):
    """
    Generate predictions for new points based on Gaussian Process uncertainty.
    Points with high uncertainty are good candidates for sampling.
    
    Parameters:
        gp: Trained Gaussian Process model
        x_scaler, y_scaler: Input data scalers
        x_values, y_values, z_values: Original data
        x_periodicity, y_periodicity: Detected periodicities
        n_points: Number of points to predict
        x_range, y_range: Range constraints
    
    Returns:
        pred_x, pred_y: Arrays of predicted point coordinates
    """
    print(f"Generating {n_points} predictions with Gaussian Process uncertainty...")
    
    # Set exploration boundaries
    if x_range is not None:
        x_min, x_max = x_range
    else:
        x_min, x_max = min(x_values) - 1.5*x_periodicity, max(x_values) + 1.5*x_periodicity
    
    if y_range is not None:
        y_min, y_max = y_range
    else:
        y_min, y_max = min(y_values) - 1.5*y_periodicity, max(y_values) + 1.5*y_periodicity
    
    # Generate a grid of candidate points
    grid_size = min(50, max(20, int(np.sqrt(n_points * 5))))
    grid_x = np.linspace(x_min, x_max, grid_size)
    grid_y = np.linspace(y_min, y_max, grid_size)
    XX, YY = np.meshgrid(grid_x, grid_y)
    grid_points_x = XX.flatten()
    grid_points_y = YY.flatten()
    
    # Scale grid points - use the same naming convention as in the training function
    X_grid = np.column_stack([grid_points_x, grid_points_y])
    X_scaled = np.column_stack([
        x_scaler.transform(grid_points_x.reshape(-1, 1)).flatten(),
        y_scaler.transform(grid_points_y.reshape(-1, 1)).flatten()
    ])
    
    try:
        # Predict with GP and get uncertainty
        _, std_pred = gp.predict(X_scaled, return_std=True)
        
        # Higher uncertainty is better for exploration
        uncertainty_scores = std_pred
        
        # Also consider distance from existing points
        distance_scores = []
        for i in range(len(grid_points_x)):
            # Calculate distances to all existing points
            distances = np.sqrt((grid_points_x[i] - x_values)**2 + 
                               (grid_points_y[i] - y_values)**2)
            min_dist = np.min(distances)
            # We want points not too close to existing data
            # but also not too far (prefer exploring the boundary)
            if min_dist < min(x_periodicity/4, y_periodicity/2):
                distance_score = 0  # Too close
            else:
                # Prefer points around 1-2 periodicities away
                optimal_dist = (x_periodicity + y_periodicity) / 2
                distance_score = np.exp(-(min_dist - optimal_dist)**2 / (optimal_dist**2))
            distance_scores.append(distance_score)
        
        distance_scores = np.array(distance_scores)
        
        # Combine uncertainty and distance scores
        combined_scores = 0.7 * uncertainty_scores + 0.3 * distance_scores
        
        # Select top points
        n_select = min(n_points, len(grid_points_x))
        top_indices = np.argsort(combined_scores)[-n_select:]
        pred_x = grid_points_x[top_indices]
        pred_y = grid_points_y[top_indices]
        
        return pred_x, pred_y
        
    except Exception as e:
        print(f"GP prediction failed: {e}")
        # Fallback to simple grid-based prediction
        n_select = min(n_points, len(grid_points_x))
        selected_indices = np.random.choice(len(grid_points_x), n_select, replace=False)
        pred_x = grid_points_x[selected_indices]
        pred_y = grid_points_y[selected_indices]
        return pred_x, pred_y
    except Exception as e:
        print(f"Warning: GP model training failed: {e}")
        # Fallback to simpler kernel
        simple_kernel = ConstantKernel(1.0) * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.1)
        gp = GaussianProcessRegressor(kernel=simple_kernel, alpha=1e-6, normalize_y=True)
        gp.fit(X_scaled, z_values)
        return gp, x_scaler, y_scaler
    

def load_data(json_file):
    """Load data from JSON file."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def export_predictions(pred_x, pred_y, output_file):
    """
    Export predicted points to a CSV file.
    
    Parameters:
        pred_x, pred_y: Arrays of predicted point coordinates
        output_file: Base output file path (will add _predictions.csv)
    """
    if len(pred_x) == 0:
        print("No points to export")
        return
    
    csv_filename = os.path.splitext(output_file)[0] + "_predictions.csv"
    with open(csv_filename, 'w') as f:
        f.write("Teff,logg\n")
        for i in range(len(pred_x)):
            f.write(f"{pred_x[i]:.2f},{pred_y[i]:.4f}\n")
    print(f"Predicted points exported to {csv_filename}")


def format_recommendations(cluster_labels, n_clusters, pred_x, pred_y, 
                         x_periodicity, y_periodicity, reconstruction_errors=None):
    """
    Format detailed recommendations based on analysis.
    
    Returns:
        recommendations: Formatted text with recommendations
    """
    recommendations = "\nDETAILED RECOMMENDATIONS FOR DATA COLLECTION:\n"
    recommendations += "="*60 + "\n\n"
    
    # Overall assessment
    if n_clusters > 0:
        recommendations += f"1. PATTERN ASSESSMENT: The data shows {n_clusters} distinct cluster pattern(s).\n"
        recommendations += "   - Recommended strategy: Focus sampling in these clustered regions.\n"
    else:
        recommendations += "1. PATTERN ASSESSMENT: No strong clustering detected in the data.\n"
        recommendations += "   - Recommended strategy: Sample according to detected periodicities.\n"
    
    # Periodicity information
    recommendations += f"\n2. DETECTED PERIODICITIES:\n"
    recommendations += f"   - Temperature (Teff): {x_periodicity:.2f}\n"
    recommendations += f"   - Surface Gravity (logg): {y_periodicity:.4f}\n"
    recommendations += f"   - For optimal sampling, collect points at these intervals\n"
    
    # If we have predicted points, group by Teff and provide ranges
    if len(pred_x) > 0:
        # Round Teff to nearest 50 for grouping
        teff_rounded = np.round(pred_x / 50) * 50
        
        # Group by rounded Teff
        teff_groups = {}
        for i in range(len(pred_x)):
            teff = teff_rounded[i]
            logg = pred_y[i]
            if teff not in teff_groups:
                teff_groups[teff] = []
            teff_groups[teff].append(logg)
        
        # Print the top 5-8 most promising Teff groups with their logg ranges
        recommendations += f"\n3. PRIORITY SAMPLING REGIONS (Top {min(8, len(teff_groups))} Temperature Bands):\n"
        
        for i, (teff, loggs) in enumerate(sorted(teff_groups.items())[:8]):
            logg_min, logg_max = min(loggs), max(loggs)
            count = len(loggs)
            recommendations += f"   • Region {i+1}: Teff ≈ {teff:.1f}, logg range: {logg_min:.2f}-{logg_max:.2f} "
            recommendations += f"({count} points recommended)\n"
        
        recommendations += f"\n   Total suggested additional points: {len(pred_x)}\n"
    
    # Add specific advice based on the reconstruction errors if available
    if reconstruction_errors is not None:
        # Find the points with highest reconstruction error
        top_error_indices = np.argsort(reconstruction_errors)[-5:]
        
        recommendations += f"\n4. REGIONS OF UNCERTAINTY (High Error Areas):\n"
        for i, idx in enumerate(top_error_indices):
            recommendations += f"   • Area {i+1}: Teff = {pred_x[idx]:.1f}, logg = {pred_y[idx]:.4f}\n"
        
        recommendations += "\n   These areas show high model uncertainty and would benefit from additional sampling.\n"
    
    # Overall summary advice
    recommendations += f"\n5. SAMPLING STRATEGY SUMMARY:\n"
    if n_clusters > 0:
        recommendations += f"   • Focus on collecting at least {max(10, len(pred_x) // 10)} additional data points\n"
        recommendations += f"   • Prioritize the clustered regions identified above\n"
        recommendations += f"   • Maintain approximate periodicity spacing within each cluster\n"
    else:
        recommendations += f"   • Sample at Teff intervals of approximately {x_periodicity:.1f}\n"
        recommendations += f"   • Sample at logg intervals of approximately {y_periodicity:.4f}\n"
        recommendations += f"   • Focus on the priority regions listed above\n"
    
    return recommendations


def main():
    parser = argparse.ArgumentParser(description='Advanced analysis and visualization of periodicities in model data.')
    parser.add_argument('json_file', help='Input JSON file with model data')
    parser.add_argument('--output', '-o', help='Output PNG file', default=None)
    parser.add_argument('--teff_min', type=float, help='Minimum Teff value for plotting', default=None)
    parser.add_argument('--teff_max', type=float, help='Maximum Teff value for plotting', default=None)
    parser.add_argument('--logg_min', type=float, help='Minimum logg value for plotting', default=None)
    parser.add_argument('--logg_max', type=float, help='Maximum logg value for plotting', default=None)
    parser.add_argument('--prediction_method', choices=['neural', 'gp', 'combined'], default='combined',
                      help='Method for predicting additional points')
    parser.add_argument('--n_predictions', type=int, default=200, 
                      help='Number of points to predict')
    args = parser.parse_args()
    
    # Set output file name based on input if not specified
    if args.output is None:
        base_name = os.path.splitext(os.path.basename(args.json_file))[0]
        args.output = f"{base_name}_advanced_analysis.png"
    
    # Set range variables
    x_range = None
    if args.teff_min is not None and args.teff_max is not None:
        x_range = (args.teff_min, args.teff_max)
    
    y_range = None
    if args.logg_min is not None and args.logg_max is not None:
        y_range = (args.logg_min, args.logg_max)
    
    # Load data
    print(f"Loading data from {args.json_file}...")
    data = load_data(args.json_file)
    
    # Extract data
    teff_values, logg_values, quality_shifted_values, z_scale_values = extract_grid_data(data)
    print(f"Extracted {len(teff_values)} data points")
    
    # Get most common z_scale for reporting
    z_scale = z_scale_values[0] if len(z_scale_values) > 0 else "N/A"
    print(f"Data z_scale: {z_scale}")
    
    # Train autoencoder for feature learning and anomaly detection
    autoencoder, data_scaler, reconstruction_errors = train_autoencoder(
        teff_values, logg_values, quality_shifted_values)
    
    # Detect periodicities
    x_periodicity, y_periodicity, x_std, y_std = detect_periodicities_wavelet(
        teff_values, logg_values, quality_shifted_values)
    
    # Find patterns/clusters in quality values
    cluster_labels, n_clusters = advanced_clustering(
        teff_values, logg_values, quality_shifted_values, reconstruction_errors)
    
    # Create Gaussian Process model
    gp, x_scaler, y_scaler = create_gaussian_process_model(
        teff_values, logg_values, quality_shifted_values, 
        x_periodicity, y_periodicity)
    
    # Generate predictions for additional data points
    pred_x = np.array([])
    pred_y = np.array([])
    
    if args.prediction_method == 'neural':
        # Neural network prediction
        pred_x, pred_y = neural_network_prediction(
            autoencoder, data_scaler, teff_values, logg_values, quality_shifted_values,
            x_periodicity, y_periodicity, n_points=args.n_predictions,
            x_range=x_range, y_range=y_range)
    
    elif args.prediction_method == 'gp':
        # Gaussian Process uncertainty prediction
        pred_x, pred_y = gp_uncertainty_prediction(
            gp, x_scaler, y_scaler, teff_values, logg_values, quality_shifted_values,
            x_periodicity, y_periodicity, n_points=args.n_predictions,
            x_range=x_range, y_range=y_range)
    
    else:  # 'combined'
        # Use both methods and combine results
        nn_pred_x, nn_pred_y = neural_network_prediction(
            autoencoder, data_scaler, teff_values, logg_values, quality_shifted_values,
            x_periodicity, y_periodicity, n_points=args.n_predictions // 2,
            x_range=x_range, y_range=y_range)
        
        gp_pred_x, gp_pred_y = gp_uncertainty_prediction(
            gp, x_scaler, y_scaler, teff_values, logg_values, quality_shifted_values,
            x_periodicity, y_periodicity, n_points=args.n_predictions // 2,
            x_range=x_range, y_range=y_range)
        
        # Combine predictions
        pred_x = np.concatenate([nn_pred_x, gp_pred_x])
        pred_y = np.concatenate([nn_pred_y, gp_pred_y])
    
    # Create visualization
    plot_analysis_results(
        teff_values, logg_values, quality_shifted_values,
        pred_x, pred_y, x_periodicity, y_periodicity,
        cluster_labels, n_clusters, z_scale, args.output,
        x_range=x_range, y_range=y_range, reconstruction_errors=reconstruction_errors)
    
    # Export predicted points
    export_predictions(pred_x, pred_y, args.output)
    
    # Print detailed recommendations
    recommendations = format_recommendations(
        cluster_labels, n_clusters, pred_x, pred_y, 
        x_periodicity, y_periodicity, reconstruction_errors)
    
    print(recommendations)
    
    # Also save recommendations to file
    rec_filename = os.path.splitext(args.output)[0] + "_recommendations.txt"
    with open(rec_filename, 'w', encoding='utf-8') as f:
        f.write(f"ADVANCED PERIODICITY ANALYSIS RESULTS\n")
        f.write(f"Data file: {args.json_file}\n")
        f.write(f"z_scale: {z_scale}\n")
        f.write(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"SUMMARY STATISTICS:\n")
        f.write(f"- Data points analyzed: {len(teff_values)}\n")
        f.write(f"- Detected Teff periodicity: {x_periodicity:.2f} ± {x_std:.2f}\n")
        f.write(f"- Detected logg periodicity: {y_periodicity:.4f} ± {y_std:.4f}\n")
        f.write(f"- Number of pattern clusters: {n_clusters}\n")
        f.write(f"- Suggested additional points: {len(pred_x)}\n")
        f.write(recommendations)
    
    print(f"Detailed recommendations saved to {rec_filename}")
    print("\nAnalysis complete!")

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

if __name__ == "__main__":
    main()