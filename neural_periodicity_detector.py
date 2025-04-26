#!/usr/bin/env python3
# enhanced_neural_periodicity_detector.py

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import logging
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import signal
from scipy.interpolate import griddata, Rbf
from scipy.spatial import ConvexHull
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF, WhiteKernel, ConstantKernel, Matern, ExpSineSquared
)
from tqdm import tqdm
from scipy import ndimage

logger = logging.getLogger("NeuralPeriodicityDetector")

# Set high-quality plot defaults
plt.style.use('seaborn-v0_8-whitegrid')
mpl.rcParams['figure.figsize'] = (12, 8)
mpl.rcParams['figure.dpi'] = 100
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 12

class Autoencoder(nn.Module):
    """
    Enhanced neural network autoencoder for anomaly detection and feature learning.
    This version includes dropout for better generalization and batch normalization.
    """
    def __init__(self, input_dim, encoding_dim=12, dropout_rate=0.2):
        super(Autoencoder, self).__init__()
        # Encoder with decreasing layer sizes and batch normalization
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, encoding_dim)
        )
        # Decoder with increasing layer sizes
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x):
        return self.encoder(x)


class NeuralPeriodicityDetector:
    """
    Advanced analysis of periodicity in model data using neural networks and 
    machine learning techniques with enhanced extrapolation capabilities.
    """
    
    def __init__(self, quality_output_dir: str, image_output_dir: str, report_output_dir: str):
        """
        Initialize the enhanced neural periodicity detector.
        
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
        os.makedirs(os.path.join(self.report_output_dir, "neural_periodicities"), exist_ok=True)
        
        # Set device for PyTorch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def run(self, quality_results_file: str) -> Optional[str]:
        """
        Run neural periodicity detection and prediction with enhanced extrapolation.
        
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
            
            if len(teff_values) < 10:
                logger.error("Not enough data points for neural periodicity analysis (need at least 10)")
                return None
            
            # Get z_scale for reporting
            z_scale = z_scale_values[0] if z_scale_values.any() else "N/A"
            
            # Train autoencoder
            logger.info("Training autoencoder for feature learning...")
            autoencoder, data_scaler, reconstruction_errors = self.train_autoencoder(
                teff_values, logg_values, quality_shifted_values)
            
            # Detect periodicities using wavelet analysis
            logger.info("Detecting periodicities with wavelet analysis...")
            x_periodicity, y_periodicity, x_std, y_std = self.detect_periodicities_wavelet(
                teff_values, logg_values, quality_shifted_values)
            
            # Find patterns/clusters
            logger.info("Finding patterns and clusters in data...")
            cluster_labels, n_clusters = self.advanced_clustering(
                teff_values, logg_values, quality_shifted_values, reconstruction_errors)
            
            # Create Gaussian Process model with periodic kernel to better capture physical patterns
            logger.info("Training enhanced Gaussian Process model...")
            gp, x_scaler, y_scaler = self.create_enhanced_gaussian_process_model(
                teff_values, logg_values, quality_shifted_values, 
                x_periodicity, y_periodicity)
            
            # Get ranges for prediction - ENHANCED: expand beyond original data range
            # Calculate a buffer based on the data range and periodicity
            x_buffer = max(x_periodicity * 2, (max(teff_values) - min(teff_values)) * 0.2)
            y_buffer = max(y_periodicity * 2, (max(logg_values) - min(logg_values)) * 0.2)
            
            x_range = (min(teff_values) - x_buffer, max(teff_values) + x_buffer)
            y_range = (min(logg_values) - y_buffer, max(logg_values) + y_buffer)
            
            logger.info(f"Expanded prediction range: Teff={x_range}, logg={y_range}")
            
            # Generate predictions with various methods
            logger.info("Generating enhanced predictions for additional data points...")
            
            # 1. Neural network prediction with extrapolation capability
            nn_pred_x, nn_pred_y = self.enhanced_neural_network_prediction(
                autoencoder, data_scaler, teff_values, logg_values, quality_shifted_values,
                x_periodicity, y_periodicity, n_points=100,
                x_range=x_range, y_range=y_range)
            
            # 2. GP prediction with uncertainty awareness
            gp_pred_x, gp_pred_y = self.enhanced_gp_prediction(
                gp, x_scaler, y_scaler, teff_values, logg_values, quality_shifted_values,
                x_periodicity, y_periodicity, n_points=100,
                x_range=x_range, y_range=y_range)
            
            # 3. Physics-based extrapolation (new method)
            phys_pred_x, phys_pred_y = self.physics_based_extrapolation(
                teff_values, logg_values, quality_shifted_values,
                x_periodicity, y_periodicity, n_points=50,
                x_range=x_range, y_range=y_range)
            
            # Combine predictions
            pred_x = np.concatenate([nn_pred_x, gp_pred_x, phys_pred_x])
            pred_y = np.concatenate([nn_pred_y, gp_pred_y, phys_pred_y])
            
            # Create visualization
            output_image = os.path.join(
                self.image_output_dir, 
                f"enhanced_neural_periodicities_{timestamp}.png"
            )
            
            self.plot_enhanced_analysis_results(
                teff_values, logg_values, quality_shifted_values,
                pred_x, pred_y, nn_pred_x, nn_pred_y, gp_pred_x, gp_pred_y, phys_pred_x, phys_pred_y,
                x_periodicity, y_periodicity,
                cluster_labels, n_clusters, z_scale, output_image,
                x_range=x_range, y_range=y_range, 
                reconstruction_errors=reconstruction_errors
            )
            
            # Export predicted points to CSV
            csv_filename = os.path.join(
                self.report_output_dir,
                "neural_periodicities",
                f"enhanced_neural_predicted_points_{timestamp}.csv"
            )
            
            if len(pred_x) > 0:
                with open(csv_filename, 'w') as f:
                    f.write("Teff,logg,source\n")
                    # Write neural network predictions
                    for i in range(len(nn_pred_x)):
                        f.write(f"{nn_pred_x[i]:.2f},{nn_pred_y[i]:.4f},neural\n")
                    # Write Gaussian process predictions    
                    for i in range(len(gp_pred_x)):
                        f.write(f"{gp_pred_x[i]:.2f},{gp_pred_y[i]:.4f},gp\n")
                    # Write physics-based predictions
                    for i in range(len(phys_pred_x)):
                        f.write(f"{phys_pred_x[i]:.2f},{phys_pred_y[i]:.4f},physics\n")
                logger.info(f"Predicted points exported to {csv_filename}")
            
            # Format detailed recommendations
            recommendations = self.format_enhanced_recommendations(
                cluster_labels, n_clusters, pred_x, pred_y, 
                nn_pred_x, nn_pred_y, gp_pred_x, gp_pred_y, phys_pred_x, phys_pred_y,
                x_periodicity, y_periodicity, reconstruction_errors)
            
            # Create report
            report_file = os.path.join(
                self.report_output_dir,
                "neural_periodicities",
                f"enhanced_neural_periodicity_analysis_{timestamp}.txt"
            )
            
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(f"ENHANCED NEURAL PERIODICITY ANALYSIS RESULTS\n")
                f.write(f"Data file: {quality_results_file}\n")
                f.write(f"z_scale: {z_scale}\n")
                f.write(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write(f"SUMMARY STATISTICS:\n")
                f.write(f"- Data points analyzed: {len(teff_values)}\n")
                f.write(f"- Detected Teff periodicity: {x_periodicity:.2f} ± {x_std:.2f}\n")
                f.write(f"- Detected logg periodicity: {y_periodicity:.4f} ± {y_std:.4f}\n")
                f.write(f"- Number of pattern clusters: {n_clusters}\n")
                f.write(f"- Suggested additional points: {len(pred_x)}\n")
                f.write(f"  - Neural network predictions: {len(nn_pred_x)}\n")
                f.write(f"  - Gaussian Process predictions: {len(gp_pred_x)}\n")
                f.write(f"  - Physics-based extrapolations: {len(phys_pred_x)}\n\n")
                f.write(f"- Expanded prediction range:\n")
                f.write(f"  - Teff: {x_range[0]:.1f} to {x_range[1]:.1f}\n")
                f.write(f"  - logg: {y_range[0]:.3f} to {y_range[1]:.3f}\n\n")
                f.write(recommendations)
            
            logger.info(f"Enhanced neural periodicity analysis report saved to {report_file}")
            
            # Also save results as JSON for use by other components
            json_results = {
                "z_scale": z_scale,
                "teff_periodicity": x_periodicity,
                "logg_periodicity": y_periodicity,
                "teff_std": x_std,
                "logg_std": y_std,
                "n_clusters": n_clusters,
                "reconstruction_errors": reconstruction_errors.tolist(),
                "cluster_labels": cluster_labels.tolist(),
                "prediction_ranges": {
                    "teff_min": float(x_range[0]),
                    "teff_max": float(x_range[1]),
                    "logg_min": float(y_range[0]),
                    "logg_max": float(y_range[1])
                },
                "predicted_points": {
                    "teff": pred_x.tolist(),
                    "logg": pred_y.tolist(),
                    "neural_network": {
                        "teff": nn_pred_x.tolist(),
                        "logg": nn_pred_y.tolist()
                    },
                    "gaussian_process": {
                        "teff": gp_pred_x.tolist(),
                        "logg": gp_pred_y.tolist()
                    },
                    "physics_based": {
                        "teff": phys_pred_x.tolist(),
                        "logg": phys_pred_y.tolist()
                    }
                }
            }
            
            json_report_file = os.path.join(
                self.report_output_dir, 
                "neural_periodicities", 
                f"enhanced_neural_periodicity_analysis_{timestamp}.json"
            )
            
            with open(json_report_file, 'w') as f:
                json.dump(json_results, f, indent=4)
            
            return json_report_file
            
        except Exception as e:
            logger.error(f"Error in enhanced neural periodicity detection: {e}")
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
    
    def train_autoencoder(self, x_values: np.ndarray, y_values: np.ndarray, 
                         z_values: np.ndarray, encoding_dim: int = 12, 
                         epochs: int = 500) -> Tuple[Autoencoder, MinMaxScaler, np.ndarray]:
        """
        Train an enhanced autoencoder for anomaly detection and feature learning.
        
        Args:
            x_values: Teff values
            y_values: logg values
            z_values: Quality values
            encoding_dim: Dimension of the encoded representation
            epochs: Number of training epochs
            
        Returns:
            model: Trained autoencoder model
            scaler: Fitted data scaler
            reconstruction_errors: Error for each input data point
        """
        # Normalize the data
        scaler = MinMaxScaler()
        data = scaler.fit_transform(np.column_stack([x_values, y_values, z_values]))
        
        # Convert to PyTorch tensors
        tensor_data = torch.FloatTensor(data).to(self.device)
        dataset = TensorDataset(tensor_data, tensor_data)
        dataloader = DataLoader(dataset, batch_size=min(32, len(data)), shuffle=True)
        
        # Initialize model
        input_dim = data.shape[1]
        model = Autoencoder(input_dim, encoding_dim, dropout_rate=0.2).to(self.device)
        
        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight decay for regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)  # Learning rate scheduler
        
        # Training loop with progress tracking
        logger.info("Training enhanced autoencoder model...")
        model.train()
        best_loss = float('inf')
        early_stop_counter = 0
        patience = 50  # Early stopping patience
        
        for epoch in tqdm(range(epochs), desc="Training epochs"):
            total_loss = 0
            for batch_features, _ in dataloader:
                batch_features = batch_features.to(self.device)
                
                # Forward pass
                outputs = model(batch_features)
                loss = criterion(outputs, batch_features)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                
            if early_stop_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
                
            # Print progress occasionally
            if (epoch+1) % 50 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # Calculate reconstruction error for each data point
        model.eval()
        with torch.no_grad():
            tensor_data = tensor_data.to(self.device)
            reconstructions = model(tensor_data)
            reconstruction_errors = torch.mean((reconstructions - tensor_data)**2, dim=1).cpu().numpy()
        
        return model, scaler, reconstruction_errors

    def detect_periodicities_wavelet(self, x_values: np.ndarray, 
                                    y_values: np.ndarray, 
                                    z_values: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Enhanced periodicity detection using wavelet transform.
        
        Args:
            x_values: Teff values
            y_values: logg values
            z_values: Quality values
            
        Returns:
            x_periodicity: Detected periodicity in x dimension
            y_periodicity: Detected periodicity in y dimension
            x_std: Standard deviation of x periodicity
            y_std: Standard deviation of y periodicity
        """
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
        
        try:
            # Interpolate to regular grid for analysis
            xi = np.linspace(min(x_values), max(x_values), 128)
            yi = np.linspace(min(y_values), max(y_values), 128)
            
            # Interpolate z values onto a regular grid
            zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic', fill_value=np.nan)
            
            # Replace NaN values
            zi = np.nan_to_num(zi, nan=np.nanmean(z_values))
            
            # For x/Teff periodicity
            # Calculate mean along rows and apply detrending to reduce edge effects
            x_mean = np.nanmean(zi, axis=0)
            x_trend = np.polyfit(np.arange(len(x_mean)), x_mean, 1)
            x_detrended = x_mean - np.polyval(x_trend, np.arange(len(x_mean)))
            
            # Use FFT on detrended data
            x_fft = np.abs(np.fft.rfft(x_detrended))
            x_freqs = np.fft.rfftfreq(len(x_mean), d=(max(xi) - min(xi)) / (len(xi) - 1))
            
            # For y/logg periodicity
            # Calculate mean along columns and detrend
            y_mean = np.nanmean(zi, axis=1)
            y_trend = np.polyfit(np.arange(len(y_mean)), y_mean, 1)
            y_detrended = y_mean - np.polyval(y_trend, np.arange(len(y_mean)))
            
            # Use FFT on detrended data
            y_fft = np.abs(np.fft.rfft(y_detrended))
            y_freqs = np.fft.rfftfreq(len(y_mean), d=(max(yi) - min(yi)) / (len(yi) - 1))
            
            # Apply window function to reduce spectral leakage
            window = np.hanning(len(x_detrended))
            x_windowed = x_detrended * window
            x_fft_windowed = np.abs(np.fft.rfft(x_windowed))
            
            window = np.hanning(len(y_detrended))
            y_windowed = y_detrended * window
            y_fft_windowed = np.abs(np.fft.rfft(y_windowed))
            
            # Find peaks in FFT power spectrum with prominence parameter
            x_peaks = signal.find_peaks(x_fft_windowed, height=np.max(x_fft_windowed)/10, prominence=np.max(x_fft_windowed)/20)[0]
            y_peaks = signal.find_peaks(y_fft_windowed, height=np.max(y_fft_windowed)/10, prominence=np.max(y_fft_windowed)/20)[0]
            
            # Calculate FFT-based periods
            x_fft_periods = []
            x_fft_powers = []
            for peak in x_peaks:
                if peak > 0 and peak < len(x_freqs):  # Skip DC component
                    period = 1.0 / x_freqs[peak]
                    if period < (max(xi) - min(xi)) / 1.5:  # Less restrictive condition
                        x_fft_periods.append(period)
                        x_fft_powers.append(x_fft_windowed[peak])
            
            y_fft_periods = []
            y_fft_powers = []
            for peak in y_peaks:
                if peak > 0 and peak < len(y_freqs):  # Skip DC component
                    period = 1.0 / y_freqs[peak]
                    if period < (max(yi) - min(yi)) / 1.5:  # Less restrictive condition
                        y_fft_periods.append(period)
                        y_fft_powers.append(y_fft_windowed[peak])
            
            # Calculate weighted average of periods based on power
            if x_fft_periods:
                x_periodicity = np.average(x_fft_periods, weights=x_fft_powers)
                x_std = np.sqrt(np.average((x_fft_periods - x_periodicity)**2, weights=x_fft_powers))
            else:
                x_periodicity = x_period_default
                x_std = x_std_default
                
            if y_fft_periods:
                y_periodicity = np.average(y_fft_periods, weights=y_fft_powers)
                y_std = np.sqrt(np.average((y_fft_periods - y_periodicity)**2, weights=y_fft_powers))
            else:
                y_periodicity = y_period_default
                y_std = y_std_default
            
            # Validate results
            if (not np.isscalar(x_periodicity) or 
                not np.isfinite(x_periodicity) or 
                x_periodicity <= 0 or 
                x_periodicity > (max(x_values) - min(x_values))):
                logger.warning("Invalid x periodicity detected, using default")
                x_periodicity = x_period_default
                x_std = x_std_default
                
            if (not np.isscalar(y_periodicity) or 
                not np.isfinite(y_periodicity) or 
                y_periodicity <= 0 or 
                y_periodicity > (max(y_values) - min(y_values))):
                logger.warning("Invalid y periodicity detected, using default")
                y_periodicity = y_period_default
                y_std = y_std_default
            
            logger.info(f"Detected x periodicity: {x_periodicity:.2f} ± {x_std:.2f}")
            logger.info(f"Detected y periodicity: {y_periodicity:.4f} ± {y_std:.4f}")
            
            return x_periodicity, y_periodicity, x_std, y_std
            
        except Exception as e:
            logger.warning(f"Advanced periodicity detection failed: {e}")
            return x_period_default, y_period_default, x_std_default, y_std_default
    
    def advanced_clustering(self, x_values: np.ndarray, y_values: np.ndarray, 
                          z_values: np.ndarray, 
                          reconstruction_errors: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
        """
        Perform advanced clustering using HDBSCAN with improved parameter tuning.
        
        Args:
            x_values: Teff values
            y_values: logg values
            z_values: Quality values
            reconstruction_errors: Optional errors from autoencoder
            
        Returns:
            cluster_labels: Cluster assignment for each data point
            n_clusters: Number of distinct clusters found
        """
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
        
        try:
            # Apply HDBSCAN clustering with adaptive parameters
            min_cluster_size = max(3, len(x_values) // 15)  # Smaller divisions for more clusters
            min_samples = max(2, len(x_values) // 30)
            
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=0.5,
                alpha=1.0  # Increased alpha for more conservative clustering
            )
            
            cluster_labels = clusterer.fit_predict(data_scaled)
            
            # Count valid clusters (excluding noise points labeled as -1)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            logger.info(f"Found {n_clusters} clusters using HDBSCAN")
            if n_clusters > 0:
                for i in range(n_clusters):
                    cluster_size = np.sum(cluster_labels == i)
                    logger.info(f"  Cluster {i}: {cluster_size} points")
            
            return cluster_labels, n_clusters
        
        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e}, falling back to simpler method")
            
            # Fallback to simple DBSCAN clustering
            from sklearn.cluster import DBSCAN
            clustering = DBSCAN(eps=0.5, min_samples=3).fit(data_scaled)
            cluster_labels = clustering.labels_
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            logger.info(f"Found {n_clusters} clusters using fallback DBSCAN")
            return cluster_labels, n_clusters
    
    def create_enhanced_gaussian_process_model(self, x_values: np.ndarray, 
                                             y_values: np.ndarray, 
                                             z_values: np.ndarray,
                                             x_periodicity: float, 
                                             y_periodicity: float) -> Tuple[GaussianProcessRegressor, 
                                                                          MinMaxScaler,
                                                                          MinMaxScaler]:
        """
        Create an enhanced Gaussian Process model with a specialized kernel
        that incorporates detected periodicities.
        
        Args:
            x_values: Teff values
            y_values: logg values
            z_values: Quality values
            x_periodicity: Detected periodicity in x dimension
            y_periodicity: Detected periodicity in y dimension
            
        Returns:
            gp: Trained Gaussian Process model
            x_scaler, y_scaler: Scalers for input normalization
        """
        # Normalize input data
        x_scaler = MinMaxScaler()
        y_scaler = MinMaxScaler()
        
        X = np.column_stack([x_values, y_values])
        X_scaled = np.column_stack([
            x_scaler.fit_transform(x_values.reshape(-1, 1)).flatten(),
            y_scaler.fit_transform(y_values.reshape(-1, 1)).flatten()
        ])
        
        try:
            # Define a more sophisticated kernel that better captures physical patterns
            # 1. RBF kernel for smooth variations
            # 2. ExpSineSquared for periodicity in both dimensions
            # 3. Matern kernel for local variations with controlled differentiability
            # 4. WhiteKernel for noise
            
            # Convert periodicities to the normalized scale
            x_period_norm = x_periodicity / (max(x_values) - min(x_values))
            y_period_norm = y_periodicity / (max(y_values) - min(y_values))
            
            # Define enhanced kernel
            kernel = (
                1.0 * RBF(length_scale=[0.5, 0.5]) + 
                1.0 * ExpSineSquared(length_scale=0.5, periodicity=x_period_norm, periodicity_bounds=(0.01, 10.0)) * 
                     RBF(length_scale=[1.0, 0.2]) +
                1.0 * ExpSineSquared(length_scale=0.5, periodicity=y_period_norm, periodicity_bounds=(0.01, 10.0)) * 
                     RBF(length_scale=[0.2, 1.0]) +
                1.0 * Matern(length_scale=[0.5, 0.5], nu=1.5) +
                WhiteKernel(noise_level=0.05)
            )
            
            # Create and train the GP model
            gp = GaussianProcessRegressor(
                kernel=kernel, 
                alpha=1e-6,
                n_restarts_optimizer=5,  # Increased for better optimization
                normalize_y=True,
                random_state=42
            )
            
            # Add small jitter to y values to avoid numerical issues
            jitter = np.random.normal(0, 1e-6, len(z_values))
            
            gp.fit(X_scaled, z_values + jitter)
            logger.info(f"Trained enhanced GP model with kernel: {gp.kernel_}")
            return gp, x_scaler, y_scaler
            
        except Exception as e:
            logger.warning(f"Enhanced GP model training failed: {e}, falling back to simpler kernel")
            
            # Fallback to even simpler kernel
            simple_kernel = ConstantKernel(1.0) * RBF(length_scale=0.5) + WhiteKernel(noise_level=0.1)
            gp = GaussianProcessRegressor(
                kernel=simple_kernel, 
                alpha=1e-6, 
                normalize_y=True,
                random_state=42
            )
            gp.fit(X_scaled, z_values)
            logger.info(f"Trained GP model with fallback kernel: {gp.kernel_}")
            return gp, x_scaler, y_scaler
    
    def enhanced_neural_network_prediction(self, model: Autoencoder, 
                                        scaler: MinMaxScaler, 
                                        x_values: np.ndarray, 
                                        y_values: np.ndarray, 
                                        z_values: np.ndarray, 
                                        x_periodicity: float, 
                                        y_periodicity: float, 
                                        n_points: int = 100, 
                                        x_range: Optional[Tuple[float, float]] = None, 
                                        y_range: Optional[Tuple[float, float]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate predictions for new points using enhanced neural network with improved extrapolation.
        
        Args:
            model: Trained autoencoder model
            scaler: Data scaler from model training
            x_values, y_values, z_values: Original data points
            x_periodicity, y_periodicity: Detected periodicities
            n_points: Number of points to predict
            x_range, y_range: Expanded range constraints for extrapolation
            
        Returns:
            pred_x, pred_y: Arrays of predicted point coordinates
        """
        logger.info(f"Generating {n_points} predictions with enhanced neural network...")
        
        # Set exploration boundaries with extended range
        if x_range is not None:
            x_min, x_max = x_range
        else:
            x_buffer = x_periodicity * 2
            x_min, x_max = min(x_values) - x_buffer, max(x_values) + x_buffer
        
        if y_range is not None:
            y_min, y_max = y_range
        else:
            y_buffer = y_periodicity * 2
            y_min, y_max = min(y_values) - y_buffer, max(y_values) + y_buffer
        
        try:
            # Generate candidate points in a grid with periodicity-based spacing
            # Use more points in extrapolation regions
            grid_size_x = max(30, int(np.ceil((x_max - x_min) / (x_periodicity / 5))))
            grid_size_y = max(30, int(np.ceil((y_max - y_min) / (y_periodicity / 5))))
            
            # Create grid of candidate points
            grid_x = np.linspace(x_min, x_max, grid_size_x)
            grid_y = np.linspace(y_min, y_max, grid_size_y)
            XX, YY = np.meshgrid(grid_x, grid_y)
            candidates_x = XX.flatten()
            candidates_y = YY.flatten()
            
            # Identify candidates in extrapolation regions (outside original data bounds)
            orig_x_min, orig_x_max = min(x_values), max(x_values)
            orig_y_min, orig_y_max = min(y_values), max(y_values)
            
            extrapolation_mask = (
                (candidates_x < orig_x_min) | (candidates_x > orig_x_max) |
                (candidates_y < orig_y_min) | (candidates_y > orig_y_max)
            )
            
            # Filter candidates for distance - extrapolation points have different criteria
            min_distances = []
            for i in range(len(candidates_x)):
                # Calculate distances to all existing points
                distances = np.sqrt((candidates_x[i] - x_values)**2 + (candidates_y[i] - y_values)**2)
                min_distances.append(np.min(distances))
            
            min_distances = np.array(min_distances)
            
            # Different distance criteria for interpolation vs extrapolation
            interp_mask = ~extrapolation_mask
            extrap_mask = extrapolation_mask
            
            # For interpolation: points should be neither too close nor too far from existing data
            interp_distance_min = min(x_periodicity/5, y_periodicity/2)
            interp_distance_max = max(x_periodicity*2, y_periodicity*10)
            valid_interp_mask = (
                interp_mask & 
                (min_distances > interp_distance_min) & 
                (min_distances < interp_distance_max)
            )
            
            # For extrapolation: points should follow a pattern extending from existing data
            # Allow points at multiples of the periodicity from the data boundary
            valid_extrap_points = []
            
            for i in np.where(extrap_mask)[0]:
                cx, cy = candidates_x[i], candidates_y[i]
                
                # Check if point is at a valid periodicity multiple from data boundary
                # For x dimension
                if cx < orig_x_min:
                    x_periods_away = (orig_x_min - cx) / x_periodicity
                elif cx > orig_x_max:
                    x_periods_away = (cx - orig_x_max) / x_periodicity
                else:
                    x_periods_away = 0
                
                # For y dimension
                if cy < orig_y_min:
                    y_periods_away = (orig_y_min - cy) / y_periodicity
                elif cy > orig_y_max:
                    y_periods_away = (cy - orig_y_max) / y_periodicity
                else:
                    y_periods_away = 0
                
                # Accept if the point is close to an integer number of periods away
                # or if it's along a natural extension of the pattern
                x_period_match = abs(x_periods_away - round(x_periods_away)) < 0.2
                y_period_match = abs(y_periods_away - round(y_periods_away)) < 0.2
                
                if (x_period_match or y_period_match) and (x_periods_away + y_periods_away < 5):
                    valid_extrap_points.append(i)
            
            valid_extrap_mask = np.zeros_like(extrap_mask, dtype=bool)
            valid_extrap_mask[valid_extrap_points] = True
            
            # Combine valid points from both interpolation and extrapolation
            valid_mask = valid_interp_mask | valid_extrap_mask
            
            filtered_x = candidates_x[valid_mask]
            filtered_y = candidates_y[valid_mask]
            filtered_extrap = extrapolation_mask[valid_mask]  # Keep track of which points are extrapolations
            
            # If not enough valid points, relax criteria
            if len(filtered_x) < n_points:
                logger.warning(f"Only {len(filtered_x)} valid candidates after filtering, relaxing criteria")
                
                # Relax criteria and try again
                interp_distance_min /= 2
                interp_distance_max *= 2
                
                valid_interp_mask = (
                    interp_mask & 
                    (min_distances > interp_distance_min) & 
                    (min_distances < interp_distance_max)
                )
                
                valid_mask = valid_interp_mask | valid_extrap_mask
                filtered_x = candidates_x[valid_mask]
                filtered_y = candidates_y[valid_mask]
                filtered_extrap = extrapolation_mask[valid_mask]
                
                if len(filtered_x) < n_points / 2:
                    logger.warning(f"Still only {len(filtered_x)} valid candidates, using random sampling")
                    indices = np.random.choice(len(candidates_x), min(n_points*2, len(candidates_x)), replace=False)
                    filtered_x = candidates_x[indices]
                    filtered_y = candidates_y[indices]
                    filtered_extrap = extrapolation_mask[indices]
            
            # Use the autoencoder to evaluate candidates
            model.eval()
            
            interestingness_scores = []
            batch_size = 1000  # Process in batches to avoid memory issues
            
            for i in range(0, len(filtered_x), batch_size):
                # Prepare batch
                batch_end = min(i + batch_size, len(filtered_x))
                batch_x = filtered_x[i:batch_end]
                batch_y = filtered_y[i:batch_end]
                batch_extrap = filtered_extrap[i:batch_end]
                
                # Create dummy z-values (we'll use the average)
                batch_z = np.ones_like(batch_x) * np.mean(z_values)
                
                # Normalize data
                batch_data = np.column_stack([batch_x, batch_y, batch_z])
                batch_data_scaled = scaler.transform(batch_data)
                batch_tensor = torch.FloatTensor(batch_data_scaled).to(self.device)
                
                # Get encoded representation
                with torch.no_grad():
                    batch_encoded = model.encode(batch_tensor).cpu().numpy()
                    
                    # Get encoded training data
                    training_data = np.column_stack([x_values, y_values, z_values])
                    training_data_scaled = scaler.transform(training_data)
                    training_tensor = torch.FloatTensor(training_data_scaled).to(self.device)
                    
                    with torch.no_grad():
                        training_encoded = model.encode(training_tensor).cpu().numpy()
                    
                    # Calculate scores differently for interpolation vs extrapolation points
                    batch_scores = []
                    for j, encoded_point in enumerate(batch_encoded):
                        distances = np.sqrt(np.sum((training_encoded - encoded_point)**2, axis=1))
                        min_dist = np.min(distances)
                        
                        if batch_extrap[j]:
                            # For extrapolation: favor points that follow the pattern but are distinct
                            # We want some novelty but not too much
                            novelty_score = np.exp(-(min_dist - 1.5)**2 / 1.0)
                            
                            # Find how many nearest training points have similar differences
                            # This rewards points that continue the pattern
                            nearest_indices = np.argsort(distances)[:5]  # Consider 5 nearest neighbors
                            pattern_score = 0
                            
                            # Check if point continues the pattern from training data
                            dx_dy_pairs = []
                            for idx in nearest_indices:
                                dx = batch_x[j] - x_values[idx]
                                dy = batch_y[j] - y_values[idx]
                                dx_dy_pairs.append((dx, dy))
                            
                            # Calculate coherence of differences (pattern continuation)
                            if len(dx_dy_pairs) > 1:
                                dx_coherence = np.std([p[0] for p in dx_dy_pairs]) / np.mean(np.abs([p[0] for p in dx_dy_pairs]))
                                dy_coherence = np.std([p[1] for p in dx_dy_pairs]) / np.mean(np.abs([p[1] for p in dx_dy_pairs]))
                                pattern_score = np.exp(-(dx_coherence + dy_coherence))
                            
                            # Combine novelty and pattern scores, weighting pattern continuation more heavily
                            score = 0.3 * novelty_score + 0.7 * pattern_score
                        else:
                            # For interpolation: use previous approach
                            # We want points that are not too close but not too far
                            score = np.exp(-(min_dist - 1.0)**2 / 0.5)
                        
                        batch_scores.append(score)
                    
                    interestingness_scores.extend(batch_scores)
            
            interestingness_scores = np.array(interestingness_scores)
            
            # Ensure a mix of interpolation and extrapolation points
            extrap_indices = np.where(filtered_extrap)[0]
            interp_indices = np.where(~filtered_extrap)[0]
            
            # Determine ratio based on available points
            if len(extrap_indices) > 0 and len(interp_indices) > 0:
                extrap_ratio = 0.6  # Favor extrapolation
                extrap_count = min(int(n_points * extrap_ratio), len(extrap_indices))
                interp_count = min(n_points - extrap_count, len(interp_indices))
                
                # Get top extrapolation points
                extrap_scores = interestingness_scores[extrap_indices]
                top_extrap = extrap_indices[np.argsort(extrap_scores)[-extrap_count:]]
                
                # Get top interpolation points
                interp_scores = interestingness_scores[interp_indices]
                top_interp = interp_indices[np.argsort(interp_scores)[-interp_count:]]
                
                # Combine
                selected_indices = np.concatenate([top_extrap, top_interp])
            else:
                # If we only have one type, just take the top n_points
                n_select = min(n_points, len(filtered_x))
                selected_indices = np.argsort(interestingness_scores)[-n_select:]
            
            pred_x = filtered_x[selected_indices]
            pred_y = filtered_y[selected_indices]
            
            logger.info(f"Generated {len(pred_x)} predictions using enhanced neural network")
            logger.info(f" - Extrapolation points: {np.sum(filtered_extrap[selected_indices])}")
            logger.info(f" - Interpolation points: {len(pred_x) - np.sum(filtered_extrap[selected_indices])}")
            
            return pred_x, pred_y
            
        except Exception as e:
            logger.error(f"Enhanced neural network prediction failed: {e}")
            # Fallback to simple prediction
            logger.info("Falling back to simple grid-based prediction")
            
            n_select = min(n_points, len(candidates_x) if 'candidates_x' in locals() else 100)
            
            # Create a simple grid if candidates weren't created successfully
            if 'candidates_x' not in locals():
                grid_x = np.linspace(x_min, x_max, 20)
                grid_y = np.linspace(y_min, y_max, 20)
                XX, YY = np.meshgrid(grid_x, grid_y)
                candidates_x = XX.flatten()
                candidates_y = YY.flatten()
            
            # Randomly select points from candidates
            if len(candidates_x) > n_select:
                selected_indices = np.random.choice(len(candidates_x), n_select, replace=False)
                pred_x = candidates_x[selected_indices]
                pred_y = candidates_y[selected_indices]
            else:
                pred_x = candidates_x
                pred_y = candidates_y
            
            logger.info(f"Generated {len(pred_x)} predictions using fallback method")
            return pred_x, pred_y

    def enhanced_gp_prediction(self, gp, x_scaler, y_scaler, x_values, y_values, z_values,
                          x_periodicity, y_periodicity, n_points=100,
                              x_range=None, y_range=None):
        """
        Use Gaussian Process with enhanced uncertainty exploration for extrapolation.
        
        Args:
            gp: Trained Gaussian Process model
            x_scaler, y_scaler: Feature scalers
            x_values, y_values: Original data points
            z_values: Quality values
            x_periodicity, y_periodicity: Detected periodicities
            n_points: Number of points to predict
            x_range, y_range: Expanded range constraints for extrapolation
            
        Returns:
            pred_x, pred_y: Arrays of predicted point coordinates
        """
        logger.info(f"Generating {n_points} predictions using enhanced Gaussian Process...")
        
        try:
            # Set exploration boundaries with expanded range
            if x_range is not None:
                x_min, x_max = x_range
            else:
                x_buffer = x_periodicity * 2
                x_min, x_max = min(x_values) - x_buffer, max(x_values) + x_buffer
            
            if y_range is not None:
                y_min, y_max = y_range
            else:
                y_buffer = y_periodicity * 2
                y_min, y_max = min(y_values) - y_buffer, max(y_values) + y_buffer
            
            # Create grid of candidate points - higher density for better coverage
            grid_size_x = max(30, int(np.ceil((x_max - x_min) / (x_periodicity / 5))))
            grid_size_y = max(30, int(np.ceil((y_max - y_min) / (y_periodicity / 5))))
            
            grid_x = np.linspace(x_min, x_max, grid_size_x)
            grid_y = np.linspace(y_min, y_max, grid_size_y)
            XX, YY = np.meshgrid(grid_x, grid_y)
            candidates_x = XX.flatten()
            candidates_y = YY.flatten()
            
            # Identify regions
            orig_x_min, orig_x_max = min(x_values), max(x_values)
            orig_y_min, orig_y_max = min(y_values), max(y_values)
            
            extrapolation_mask = (
                (candidates_x < orig_x_min) | (candidates_x > orig_x_max) |
                (candidates_y < orig_y_min) | (candidates_y > orig_y_max)
            )
            
            # Calculate distance to nearest data point for all candidates
            min_distances = []
            for i in range(len(candidates_x)):
                distances = np.sqrt((candidates_x[i] - x_values)**2 + 
                                  (candidates_y[i] - y_values)**2)
                min_distances.append(np.min(distances))
            
            min_distances = np.array(min_distances)
            
            # Different filtering strategy for interpolation vs extrapolation
            interp_mask = ~extrapolation_mask
            extrap_mask = extrapolation_mask
            
            # For interpolation: not too close to existing points
            distance_threshold_min = min(x_periodicity/4, y_periodicity/2)
            valid_interp = interp_mask & (min_distances > distance_threshold_min)
            
            # For extrapolation: favor points at multiples of periodicity from data boundary
            valid_extrap_indices = []
            
            for i in np.where(extrap_mask)[0]:
                cx, cy = candidates_x[i], candidates_y[i]
                
                # Check if point is at a valid periodicity multiple from data boundary
                # Simplified version that just checks distance to boundary in units of periodicity
                x_dist = min(abs(cx - orig_x_min), abs(cx - orig_x_max))
                y_dist = min(abs(cy - orig_y_min), abs(cy - orig_y_max))
                
                x_periods = x_dist / x_periodicity
                y_periods = y_dist / y_periodicity
                
                # Accept if close to integer periods and not too far out
                x_valid = abs(x_periods - round(x_periods)) < 0.2 and x_periods < 5
                y_valid = abs(y_periods - round(y_periods)) < 0.2 and y_periods < 5
                
                if x_valid or y_valid:
                    valid_extrap_indices.append(i)
            
            valid_extrap = np.zeros_like(extrap_mask, dtype=bool)
            valid_extrap[valid_extrap_indices] = True
            
            # Combine valid candidates
            valid_mask = valid_interp | valid_extrap
            
            if np.sum(valid_mask) < n_points / 2:
                logger.warning("Few valid candidates after filtering, relaxing criteria")
                # Fallback to distance-based filter only
                valid_mask = min_distances > distance_threshold_min
            
            filtered_x = candidates_x[valid_mask]
            filtered_y = candidates_y[valid_mask]
            filtered_extrap = extrapolation_mask[valid_mask]
            
            # Prepare for GP prediction
            X_grid = np.column_stack([
                x_scaler.transform(filtered_x.reshape(-1, 1)).flatten(),
                y_scaler.transform(filtered_y.reshape(-1, 1)).flatten()
            ])
            
            # Get GP prediction and uncertainty
            y_mean, y_std = gp.predict(X_grid, return_std=True)
            
            # Calculate selection scores differently for interpolation vs extrapolation
            selection_scores = np.zeros(len(filtered_x))
            
            # For interpolation points: favor high uncertainty
            interp_indices = np.where(~filtered_extrap)[0]
            if len(interp_indices) > 0:
                selection_scores[interp_indices] = y_std[interp_indices]
            
            # For extrapolation points: balance uncertainty with pattern continuation
            extrap_indices = np.where(filtered_extrap)[0]
            if len(extrap_indices) > 0:
                # Base score is uncertainty
                extrap_scores = y_std[extrap_indices]
                
                # Adjust based on periodicity pattern
                for i, idx in enumerate(extrap_indices):
                    cx, cy = filtered_x[idx], filtered_y[idx]
                    
                    # Check alignment with periodicity grid from center of existing data
                    center_x = (orig_x_min + orig_x_max) / 2
                    center_y = (orig_y_min + orig_y_max) / 2
                    
                    # Distance from ideal grid lines
                    x_phase = ((cx - center_x) % x_periodicity) / x_periodicity
                    y_phase = ((cy - center_y) % y_periodicity) / y_periodicity
                    
                    # Points closer to ideal grid lines get a bonus (smaller phase is better)
                    x_alignment = min(x_phase, 1 - x_phase)  # Distance to nearest grid line
                    y_alignment = min(y_phase, 1 - y_phase)
                    
                    # Transform to [0,1] where 1 means perfect alignment
                    x_alignment_score = 1 - x_alignment
                    y_alignment_score = 1 - y_alignment
                    
                    # Combined alignment score (geometric mean)
                    alignment_score = np.sqrt(x_alignment_score * y_alignment_score)
                    
                    # Final score combines uncertainty and alignment
                    extrap_scores[i] = extrap_scores[i] * (0.5 + 0.5 * alignment_score)
                
                selection_scores[extrap_indices] = extrap_scores
            
            # Select points with mix of interpolation and extrapolation
            if len(extrap_indices) > 0 and len(interp_indices) > 0:
                # Target ratio of extrapolation points
                extrap_ratio = 0.7  # Favor extrapolation
                extrap_count = min(int(n_points * extrap_ratio), len(extrap_indices))
                interp_count = min(n_points - extrap_count, len(interp_indices))
                
                # Select top scoring points from each group
                top_extrap = extrap_indices[np.argsort(selection_scores[extrap_indices])[-extrap_count:]]
                top_interp = interp_indices[np.argsort(selection_scores[interp_indices])[-interp_count:]]
                
                # Combine selections
                selected_indices = np.concatenate([top_extrap, top_interp])
            else:
                # If we have only one type, just select top scoring points
                n_select = min(n_points, len(filtered_x))
                selected_indices = np.argsort(selection_scores)[-n_select:]
            
            pred_x = filtered_x[selected_indices]
            pred_y = filtered_y[selected_indices]
            
            logger.info(f"Generated {len(pred_x)} predictions using enhanced GP")
            logger.info(f" - Extrapolation points: {np.sum(filtered_extrap[selected_indices])}")
            logger.info(f" - Interpolation points: {len(pred_x) - np.sum(filtered_extrap[selected_indices])}")
            
            return pred_x, pred_y
            
        except Exception as e:
            logger.error(f"Enhanced GP prediction failed: {e}")
            logger.info("Falling back to simple grid-based prediction")
            
            # Use grid-based approach as fallback
            n_select = min(n_points, len(candidates_x) if 'candidates_x' in locals() else 100)
            
            # Create a simple grid if candidates weren't created successfully
            if 'candidates_x' not in locals():
                grid_x = np.linspace(x_min, x_max, 20)
                grid_y = np.linspace(y_min, y_max, 20)
                XX, YY = np.meshgrid(grid_x, grid_y)
                candidates_x = XX.flatten()
                candidates_y = YY.flatten()
            
            # Randomly select points from candidates
            if len(candidates_x) > n_select:
                selected_indices = np.random.choice(len(candidates_x), n_select, replace=False)
                pred_x = candidates_x[selected_indices]
                pred_y = candidates_y[selected_indices]
            else:
                pred_x = candidates_x
                pred_y = candidates_y
            
            logger.info(f"Generated {len(pred_x)} predictions using fallback method")
            return pred_x, pred_y

    def physics_based_extrapolation(self, x_values, y_values, z_values, 
                                 x_periodicity, y_periodicity, n_points=50,
                                 x_range=None, y_range=None):
        """
        Generate extrapolated points based on physical properties of stellar atmospheres.
        This method uses theoretical relationships between Teff, logg, and quality.
        
        Args:
            x_values, y_values: Original data points (Teff, logg)
            z_values: Quality values
            x_periodicity, y_periodicity: Detected periodicities
            n_points: Number of points to generate
            x_range, y_range: Expanded range constraints for extrapolation
            
        Returns:
            pred_x, pred_y: Arrays of predicted point coordinates
        """
        logger.info(f"Generating {n_points} predictions using physics-based extrapolation...")
        
        try:
            # Set exploration boundaries
            if x_range is not None:
                x_min, x_max = x_range
            else:
                x_buffer = x_periodicity * 2
                x_min, x_max = min(x_values) - x_buffer, max(x_values) + x_buffer
            
            if y_range is not None:
                y_min, y_max = y_range
            else:
                y_buffer = y_periodicity * 2
                y_min, y_max = min(y_values) - y_buffer, max(y_values) + y_buffer
                
            # Define the boundaries of the original data
            orig_x_min, orig_x_max = min(x_values), max(x_values)
            orig_y_min, orig_y_max = min(y_values), max(y_values)
            
            # Create candidates for extrapolation only
            extrapolated_points = []
            
            # 1. Generate points along stellar evolution tracks
            # In Teff-logg space, stellar evolution typically follows certain tracks
            # We'll use simplified relationships
            
            # a) Main sequence relationship: logg ~ 4.0 - 0.0005 * (Teff - 5800)
            # Generate points along this track
            ms_teff_values = np.linspace(x_min, x_max, 20)
            for teff in ms_teff_values:
                # Skip points inside the original data range
                if orig_x_min <= teff <= orig_x_max:
                    continue
                
                # Calculate logg based on main sequence relation
                logg = 4.0 - 0.0005 * (teff - 5800)
                
                # Only add if within y_range
                if y_min <= logg <= y_max:
                    # Only add if outside original data boundaries
                    if not (orig_x_min <= teff <= orig_x_max and orig_y_min <= logg <= orig_y_max):
                        extrapolated_points.append((teff, logg))
            
            # b) Giant branch relationship: logg decreases more rapidly with decreasing Teff
            # For stars with Teff < 5500K, logg tends to decrease more steeply
            gb_teff_values = np.linspace(x_min, min(5500, x_max), 15)
            for teff in gb_teff_values:
                # Skip points inside the original data range
                if orig_x_min <= teff <= orig_x_max:
                    continue
                
                # Calculate logg based on giant branch relation
                logg = 2.5 - 0.001 * (teff - 4500)
                
                # Only add if within y_range
                if y_min <= logg <= y_max:
                    # Only add if outside original data boundaries
                    if not (orig_x_min <= teff <= orig_x_max and orig_y_min <= logg <= orig_y_max):
                        extrapolated_points.append((teff, logg))
            
            # 2. Generate points at multiples of the detected periodicities from the data boundaries
            # Use a grid-like approach, extending the pattern outward
            
            # Find centers of each dimension
            x_center = (orig_x_min + orig_x_max) / 2
            y_center = (orig_y_min + orig_y_max) / 2
            
            # Generate grid points extending outward
            for i in range(-5, 6):
                for j in range(-5, 6):
                    # Skip i=j=0 (center point)
                    if i == 0 and j == 0:
                        continue
                    
                    # Calculate grid point
                    teff = x_center + i * x_periodicity
                    logg = y_center + j * y_periodicity
                    
                    # Check if it's in range
                    if x_min <= teff <= x_max and y_min <= logg <= y_max:
                        # Only add if outside original data boundaries
                        if not (orig_x_min <= teff <= orig_x_max and orig_y_min <= logg <= orig_y_max):
                            extrapolated_points.append((teff, logg))
            
            # 3. Generate points around the Hull of the data
            # Create a convex hull around the original data
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(np.column_stack([x_values, y_values]))
                hull_vertices = np.column_stack([x_values, y_values])[hull.vertices]
                
                # Generate points slightly outside the hull boundary
                for i in range(len(hull_vertices)):
                    v1 = hull_vertices[i]
                    v2 = hull_vertices[(i + 1) % len(hull_vertices)]
                    
                    # Calculate midpoint and normal vector
                    midpoint = (v1 + v2) / 2
                    direction = v2 - v1
                    normal = np.array([-direction[1], direction[0]])  # Perpendicular to direction
                    normal = normal / np.linalg.norm(normal)  # Normalize
                    
                    # Create points at distances outside the hull
                    for dist in [0.5, 1.0, 1.5]:
                        scale_factor = max(x_periodicity, y_periodicity) * dist
                        new_point = midpoint + normal * scale_factor
                        
                        # Check if in range and outside original data
                        teff, logg = new_point
                        if (x_min <= teff <= x_max and y_min <= logg <= y_max and
                            not (orig_x_min <= teff <= orig_x_max and orig_y_min <= logg <= orig_y_max)):
                            extrapolated_points.append((teff, logg))
            except Exception as e:
                logger.warning(f"Convex hull generation failed: {e}")
            
            # Convert to numpy arrays
            if extrapolated_points:
                extrapolated_points = np.array(extrapolated_points)
                pred_x = extrapolated_points[:, 0]
                pred_y = extrapolated_points[:, 1]
                
                # If we have too many points, select a representative subset
                if len(pred_x) > n_points:
                    # Use farthest point sampling for diversity
                    selected_indices = self._farthest_point_sampling(
                        np.column_stack([pred_x, pred_y]), n_points)
                    pred_x = pred_x[selected_indices]
                    pred_y = pred_y[selected_indices]
            else:
                # Fallback if no points were generated
                logger.warning("No physics-based extrapolation points generated, falling back to grid sampling")
                grid_x = np.linspace(x_min, x_max, int(np.sqrt(n_points)))
                grid_y = np.linspace(y_min, y_max, int(np.sqrt(n_points)))
                XX, YY = np.meshgrid(grid_x, grid_y)
                candidates_x = XX.flatten()
                candidates_y = YY.flatten()
                
                # Keep only points outside the original data range
                extrap_mask = (
                    (candidates_x < orig_x_min) | (candidates_x > orig_x_max) |
                    (candidates_y < orig_y_min) | (candidates_y > orig_y_max)
                )
                pred_x = candidates_x[extrap_mask]
                pred_y = candidates_y[extrap_mask]
                
                # If still too many points, randomly select subset
                if len(pred_x) > n_points:
                    indices = np.random.choice(len(pred_x), n_points, replace=False)
                    pred_x = pred_x[indices]
                    pred_y = pred_y[indices]
            
            logger.info(f"Generated {len(pred_x)} predictions using physics-based extrapolation")
            return pred_x, pred_y
            
        except Exception as e:
            logger.error(f"Physics-based extrapolation failed: {e}")
            logger.info("Falling back to simple grid-based prediction")
            
            # Simple grid fallback
            grid_x = np.linspace(x_min, x_max, 15)
            grid_y = np.linspace(y_min, y_max, 15)
            XX, YY = np.meshgrid(grid_x, grid_y)
            candidates_x = XX.flatten()
            candidates_y = YY.flatten()
            
            # Keep only points outside original data range
            orig_x_min, orig_x_max = min(x_values), max(x_values)
            orig_y_min, orig_y_max = min(y_values), max(y_values)
            
            extrap_mask = (
                (candidates_x < orig_x_min) | (candidates_x > orig_x_max) |
                (candidates_y < orig_y_min) | (candidates_y > orig_y_max)
            )
            
            pred_x = candidates_x[extrap_mask]
            pred_y = candidates_y[extrap_mask]
            
            # If too many points, randomly select subset
            if len(pred_x) > n_points:
                indices = np.random.choice(len(pred_x), n_points, replace=False)
                pred_x = pred_x[indices]
                pred_y = pred_y[indices]
                
            logger.info(f"Generated {len(pred_x)} predictions using fallback method")
            return pred_x, pred_y
    
    def _farthest_point_sampling(self, points, n_samples):
        """
        Select a diverse subset of points using farthest point sampling.
        
        Args:
            points: Array of points (N, D)
            n_samples: Number of points to select
            
        Returns:
            indices: Indices of selected points
        """
        n_samples = min(n_samples, len(points))
        selected_indices = [np.random.randint(len(points))]  # Start with random point
        
        # Calculate pairwise distances (inefficient but simple)
        distances = np.zeros((len(points), len(points)))
        for i in range(len(points)):
            for j in range(i+1, len(points)):
                distances[i, j] = np.linalg.norm(points[i] - points[j])
                distances[j, i] = distances[i, j]
        
        # Select remaining points
        for _ in range(n_samples - 1):
            # Find distances from all points to selected set
            min_dists = np.min(distances[selected_indices][:, np.arange(len(points))], axis=0)
            
            # Select farthest point
            new_idx = np.argmax(min_dists)
            selected_indices.append(new_idx)
        
        return selected_indices

    def plot_enhanced_analysis_results(self, x_values, y_values, z_values,
                       pred_x, pred_y, nn_pred_x, nn_pred_y, gp_pred_x, gp_pred_y, 
                       phys_pred_x, phys_pred_y, x_periodicity, y_periodicity,
                       cluster_labels, n_clusters, z_scale, output_file,
                       x_range=None, y_range=None, reconstruction_errors=None):
        """
        Create enhanced visualization of neural periodicity analysis results with multiple prediction methods.
        
        Args:
            x_values: Array of x-values (Teff)
            y_values: Array of y-values (logg)
            z_values: Array of z-values (quality)
            pred_x, pred_y: Combined arrays of predicted point coordinates
            nn_pred_x, nn_pred_y: Neural network predictions
            gp_pred_x, gp_pred_y: Gaussian Process predictions
            phys_pred_x, phys_pred_y: Physics-based predictions
            x_periodicity: Detected x periodicity
            y_periodicity: Detected y periodicity
            cluster_labels: Array of cluster labels
            n_clusters: Number of clusters
            z_scale: Z-scale value
            output_file: Path to save the output image
            x_range, y_range: Range for the plot
            reconstruction_errors: Optional autoencoder reconstruction errors
        """
        plt.figure(figsize=(16, 12))
        
        # Create a custom colormap for quality heatmap
        colors = ['darkviolet', 'navy', 'teal', 'green', 'yellowgreen', 'yellow']
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # Calculate bounds for the plot
        if x_range and y_range:
            x_min, x_max = x_range
            y_min, y_max = y_range
        else:
            # Calculate from data and predictions
            if len(pred_x) > 0:
                x_min = min(min(x_values), min(pred_x))
                x_max = max(max(x_values), max(pred_x))
                y_min = min(min(y_values), min(pred_y))
                y_max = max(max(y_values), max(pred_y))
            else:
                x_min, x_max = min(x_values), max(x_values)
                y_min, y_max = min(y_values), max(y_values)
        
        # Add some padding
        x_range_size = x_max - x_min
        y_range_size = y_max - y_min
        x_min -= 0.05 * x_range_size
        x_max += 0.05 * x_range_size
        y_min -= 0.05 * y_range_size
        y_max += 0.05 * y_range_size
        
        # Create a grid for the contour plot
        xi = np.linspace(min(x_values), max(x_values), 100)
        yi = np.linspace(min(y_values), max(y_values), 100)
        zi = griddata((x_values, y_values), z_values, (xi[None,:], yi[:,None]), method='cubic')
        
        # Replace NaN values with the mean of non-NaN values
        if np.any(np.isnan(zi)):
            zi = np.nan_to_num(zi, nan=np.nanmean(zi[~np.isnan(zi)]))
        
        # Plot the contour
        contour = plt.contourf(xi, yi, zi, 100, cmap=cmap, extend='both')
        plt.colorbar(contour, label='Quality (Shifted)')
        
        # Plot the original data points with size proportional to reconstruction error if available
        if reconstruction_errors is not None:
            # Normalize reconstruction errors for point sizing
            norm_errors = (reconstruction_errors - np.min(reconstruction_errors))
            if np.max(norm_errors) > 0:
                norm_errors = norm_errors / np.max(norm_errors)
            
            # Size range from 30 to 150 based on reconstruction error
            sizes = 30 + norm_errors * 120
            
            # Create a scatter plot with size based on reconstruction error
            scatter = plt.scatter(x_values, y_values, marker='o', s=sizes,
                            c=z_values, cmap='plasma', 
                            edgecolor='white', linewidth=1.0,
                            label='Original Data Points (size = reconstruction error)')
        else:
            # If no reconstruction errors, use uniform size
            plt.scatter(x_values, y_values, marker='o', s=60, 
                    edgecolor='white', facecolor='black', linewidth=1.5,
                    label='Original Data Points')
        
        # Create a semi-transparent rectangle showing the original data range
        orig_x_min, orig_x_max = min(x_values), max(x_values)
        orig_y_min, orig_y_max = min(y_values), max(y_values)
        
        plt.fill_between([orig_x_min, orig_x_max], [orig_y_min, orig_y_min], 
                       [orig_y_max, orig_y_max], color='gray', alpha=0.1)
        
        # Add border around the original data range
        plt.plot([orig_x_min, orig_x_max, orig_x_max, orig_x_min, orig_x_min],
               [orig_y_min, orig_y_min, orig_y_max, orig_y_max, orig_y_min],
               'k--', alpha=0.5, linewidth=1, label='Original Data Range')
        
        # Highlight detected clusters if they exist
        unique_labels = set(cluster_labels)
        if n_clusters > 0:  # If we have more than just noise (-1)
            for label in unique_labels:
                if label == -1:  # Skip noise points
                    continue
                mask = cluster_labels == label
                plt.scatter(x_values[mask], y_values[mask], marker='*', s=150,
                        edgecolor='white', linewidth=1.5, 
                        label=f'Cluster {label}')
        
        # Plot predictions from different methods with distinct markers
        if len(nn_pred_x) > 0:
            # Neural network predictions - 'x' marker
            sample_size = min(100, len(nn_pred_x))
            sample_indices = np.random.choice(len(nn_pred_x), sample_size, replace=False)
            plt.scatter(nn_pred_x[sample_indices], nn_pred_y[sample_indices], 
                    c='red', marker='x', s=50, 
                    label='Neural Network Predictions')
        
        if len(gp_pred_x) > 0:
            # GP predictions - 'diamond' marker
            sample_size = min(100, len(gp_pred_x))
            sample_indices = np.random.choice(len(gp_pred_x), sample_size, replace=False)
            plt.scatter(gp_pred_x[sample_indices], gp_pred_y[sample_indices], 
                    c='magenta', marker='d', s=50, 
                    label='Gaussian Process Predictions')
        
        if len(phys_pred_x) > 0:
            # Physics-based predictions - 'triangle' marker
            sample_size = min(100, len(phys_pred_x))
            sample_indices = np.random.choice(len(phys_pred_x), sample_size, replace=False)
            plt.scatter(phys_pred_x[sample_indices], phys_pred_y[sample_indices], 
                    c='orange', marker='^', s=50, 
                    label='Physics-Based Predictions')
        
        # Create density visualization for all predictions combined
        if len(pred_x) > 0:
            # Create a 2D histogram of predicted points for density visualization
            pred_hist, xedges, yedges = np.histogram2d(
                pred_x, pred_y, 
                bins=[np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50)]
            )
            
            # Apply Gaussian smoothing
            pred_hist_smooth = ndimage.gaussian_filter(pred_hist, sigma=1.0)
            
            # Plot contour lines to show prediction density
            X_grid, Y_grid = np.meshgrid(
                (xedges[:-1] + xedges[1:]) / 2, 
                (yedges[:-1] + yedges[1:]) / 2
            )
            plt.contour(X_grid, Y_grid, pred_hist_smooth.T, 
                    levels=5, colors='blue', alpha=0.7, 
                    linestyles='dashed', linewidths=1)
        
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
        
        # Set labels and title
        plt.xlabel('Effective Temperature (Teff)', fontsize=14)
        plt.ylabel('Surface Gravity (log g)', fontsize=14)
        plt.title(f'Enhanced Neural Periodicity Analysis for Stellar Atmosphere Models (z_scale={z_scale})\n'
                f'Detected Periodicities: Teff={x_periodicity:.2f}, logg={y_periodicity:.4f}', 
                fontsize=16)
        
        # Add text box with analysis info
        info_str = f'Detected Teff Periodicity: {x_periodicity:.2f}\n'
        info_str += f'Detected logg Periodicity: {y_periodicity:.4f}\n'
        info_str += f'Number of Pattern Clusters: {n_clusters}\n'
        info_str += f'Total Suggested Points: {len(pred_x)}\n'
        info_str += f'  - Neural Network: {len(nn_pred_x)}\n'
        info_str += f'  - Gaussian Process: {len(gp_pred_x)}\n'
        info_str += f'  - Physics-Based: {len(phys_pred_x)}'
        
        plt.annotate(info_str, xy=(0.02, 0.02), xycoords='axes fraction',
                fontsize=10, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
        
        # Add legend with smaller font
        plt.legend(loc='upper right', fontsize=10, ncol=2)
        
        # Set axis limits
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Enhanced neural periodicity analysis plot saved to {output_file}")

    def format_enhanced_recommendations(self, cluster_labels, n_clusters, pred_x, pred_y, 
                                nn_pred_x, nn_pred_y, gp_pred_x, gp_pred_y, 
                                phys_pred_x, phys_pred_y,
                                x_periodicity, y_periodicity, reconstruction_errors,
                                x_values=None, y_values=None):
        """
        Format detailed recommendations based on enhanced analysis results.
        
        Args:
            cluster_labels: Array of cluster labels
            n_clusters: Number of distinct clusters found
            pred_x, pred_y: Combined arrays of predicted coordinates
            nn_pred_x, nn_pred_y: Neural network predictions
            gp_pred_x, gp_pred_y: Gaussian Process predictions
            phys_pred_x, phys_pred_y: Physics-based predictions
            x_periodicity: Detected periodicity in x dimension
            y_periodicity: Detected periodicity in y dimension
            reconstruction_errors: Autoencoder reconstruction errors
            
        Returns:
            String containing formatted recommendations
        """
        recommendations = "\n\nENHANCED RECOMMENDATIONS:\n"
        recommendations += "========================\n\n"
        
        # Recommendations based on clusters
        if n_clusters > 0:
            recommendations += f"1. CLUSTER-BASED SAMPLING STRATEGY:\n"
            recommendations += f"   The data reveals {n_clusters} distinct pattern clusters. We recommend:\n"
            
            # Find centers of clusters
            cluster_centers = {}
            for i in range(n_clusters):
                mask = cluster_labels == i
                if np.any(mask):
                    cluster_centers[i] = (f"Cluster {i}", len(np.where(mask)[0]))
            
            for i, (name, size) in cluster_centers.items():
                recommendations += f"   - {name}: Contains {size} points\n"
        
        # Recommendations based on periodicities
        recommendations += f"\n2. PERIODICITY-BASED SAMPLING STRATEGY:\n"
        recommendations += f"   The detected periodicities suggest a grid-based sampling approach:\n"
        recommendations += f"   - Sample at Teff intervals of {x_periodicity:.1f}\n"
        recommendations += f"   - Sample at logg intervals of {y_periodicity:.4f}\n"
        
        # Recommendations by prediction method
        recommendations += f"\n3. ENHANCED PREDICTION STRATEGY:\n"
        recommendations += f"   Based on our multi-method approach, we recommend prioritizing:\n"
        
        # Neural network predictions
        if len(nn_pred_x) > 0:
            # Group by Teff values (rounded)
            nn_teff_grouped = {}
            for i in range(len(nn_pred_x)):
                teff_rounded = round(nn_pred_x[i] / 100) * 100  # Round to nearest 100
                if teff_rounded not in nn_teff_grouped:
                    nn_teff_grouped[teff_rounded] = []
                nn_teff_grouped[teff_rounded].append(nn_pred_y[i])
            
            recommendations += f"   A. Neural Network Priority Regions:\n"
            for i, (teff, loggs) in enumerate(sorted(nn_teff_grouped.items())[:3]):  # Top 3 regions
                if loggs:
                    logg_min, logg_max = min(loggs), max(loggs)
                    recommendations += f"      - Region {i+1}: Teff = {teff:.1f}, logg range: {logg_min:.2f}-{logg_max:.2f}\n"
        
        # Gaussian Process predictions
        if len(gp_pred_x) > 0:
            # Group by Teff values (rounded)
            gp_teff_grouped = {}
            for i in range(len(gp_pred_x)):
                teff_rounded = round(gp_pred_x[i] / 100) * 100  # Round to nearest 100
                if teff_rounded not in gp_teff_grouped:
                    gp_teff_grouped[teff_rounded] = []
                gp_teff_grouped[teff_rounded].append(gp_pred_y[i])
            
            recommendations += f"   B. Gaussian Process Uncertainty Regions:\n"
            for i, (teff, loggs) in enumerate(sorted(gp_teff_grouped.items())[:3]):  # Top 3 regions
                if loggs:
                    logg_min, logg_max = min(loggs), max(loggs)
                    recommendations += f"      - Region {i+1}: Teff = {teff:.1f}, logg range: {logg_min:.2f}-{logg_max:.2f}\n"
        
        # Physics-based predictions
        if len(phys_pred_x) > 0:
            # Group by Teff values (rounded)
            phys_teff_grouped = {}
            for i in range(len(phys_pred_x)):
                teff_rounded = round(phys_pred_x[i] / 100) * 100  # Round to nearest 100
                if teff_rounded not in phys_teff_grouped:
                    phys_teff_grouped[teff_rounded] = []
                phys_teff_grouped[teff_rounded].append(phys_pred_y[i])
            
            recommendations += f"   C. Physics-Based Extrapolation Regions:\n"
            for i, (teff, loggs) in enumerate(sorted(phys_teff_grouped.items())[:3]):  # Top 3 regions
                if loggs:
                    logg_min, logg_max = min(loggs), max(loggs)
                    recommendations += f"      - Region {i+1}: Teff = {teff:.1f}, logg range: {logg_min:.2f}-{logg_max:.2f}\n"
                    
        # Add anomaly detection recommendations based on reconstruction errors
        if reconstruction_errors is not None and len(reconstruction_errors) > 0:
            threshold = np.mean(reconstruction_errors) + 1.5 * np.std(reconstruction_errors)
            anomaly_indices = np.where(reconstruction_errors > threshold)[0]
            
            if len(anomaly_indices) > 0:
                recommendations += f"\n4. ANOMALY INVESTIGATION RECOMMENDATIONS:\n"
                recommendations += f"   We detected {len(anomaly_indices)} potential anomalies with high reconstruction error.\n"
                recommendations += f"   Top anomalies to investigate:\n"
                
                # Sort anomalies by reconstruction error and show top 5
                sorted_indices = anomaly_indices[np.argsort(reconstruction_errors[anomaly_indices])[-5:]]
                for i, idx in enumerate(sorted_indices):
                    error = reconstruction_errors[idx]
                    recommendations += f"      - Anomaly {i+1}: Teff={x_values[idx]:.1f}, logg={y_values[idx]:.4f} (error: {error:.6f})\n"
        
        # Final priority sampling recommendations
        recommendations += f"\n5. PRIORITY SAMPLING RECOMMENDATIONS:\n"
        recommendations += f"   Based on our comprehensive analysis, we recommend the following sampling priorities:\n"
        
        # Combine all prediction methods and find the highest density regions
        combined_teff_grouped = {}
        
        # Process all points
        for x, y in zip(pred_x, pred_y):
            teff_rounded = round(x / 100) * 100
            if teff_rounded not in combined_teff_grouped:
                combined_teff_grouped[teff_rounded] = []
            combined_teff_grouped[teff_rounded].append(y)
        
        # Find regions with highest point density
        region_densities = {teff: len(loggs) for teff, loggs in combined_teff_grouped.items()}
        top_regions = sorted(region_densities.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (teff, count) in enumerate(top_regions):
            loggs = combined_teff_grouped[teff]
            logg_min, logg_max = min(loggs), max(loggs)
            recommendations += f"   - Priority {i+1}: Teff = {teff:.1f}, logg range: {logg_min:.2f}-{logg_max:.2f} ({count} predictions)\n"
        
        # Overall sampling strategy
        recommendations += f"\n6. OVERALL SAMPLING STRATEGY:\n"
        recommendations += f"   For efficient model space exploration, we recommend:\n"
        recommendations += f"   - Collect at least {max(20, len(pred_x) // 5)} additional data points based on this analysis\n"
        recommendations += f"   - Focus 60% of sampling on extrapolation regions outside the current data range\n"
        recommendations += f"   - Sample at intervals matching the detected periodicities: Teff={x_periodicity:.1f}, logg={y_periodicity:.4f}\n"
        recommendations += f"   - Prioritize regions identified in the top 5 priority regions above\n"
        
        return recommendations