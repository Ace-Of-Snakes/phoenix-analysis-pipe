#!/usr/bin/env python3
# heatmap_optimizer.py

import os
import json
import numpy as np
import logging
import datetime
import warnings
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy.interpolate import griddata
from scipy.optimize import (
    minimize, 
    differential_evolution,
    dual_annealing
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

logger = logging.getLogger("HeatmapMinimumFinder")

class HeatmapMinimumFinder:
    """
    Find local and global minima in the heatmap of PHOENIX model quality values.
    Based on HGMF_multithreaded.py.
    """
    
    def __init__(self, quality_output_dir: str, image_output_dir: str, 
                 report_output_dir: str, threads: int = 4):
        """
        Initialize the heatmap optimizer.
        
        Args:
            quality_output_dir: Directory containing quality calculation results
            image_output_dir: Directory for storing output images
            report_output_dir: Directory for storing reports
            threads: Number of threads to use
        """
        self.quality_output_dir = quality_output_dir
        self.image_output_dir = image_output_dir
        self.report_output_dir = report_output_dir
        self.threads = threads
        
        # Ensure output directories exist
        os.makedirs(self.image_output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.report_output_dir, "minima"), exist_ok=True)
        
        # Attributes to be set during execution
        self.json_file = None
        self.data = None
        self.points = []
        self.values = []
        self.z_scale = None
        self.bounds = None
        self.grid_x = None
        self.grid_y = None
        self.grid_z = None
        self.de_points = []  # Differential evolution points
        self.da_points = []  # Dual annealing points
    
    def run(self, quality_results_file: str) -> Optional[str]:
        """
        Run the heatmap optimization to find global minimum.
        
        Args:
            quality_results_file: Path to JSON file with quality calculation results
            
        Returns:
            Path to the optimization results report file, or None if failed
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.json_file = quality_results_file
        
        try:
            # Load quality data
            self.data = self.load_data()
            if not self.data:
                logger.error("Failed to load quality data")
                return None
            
            # Extract points and values
            self.extract_points_values()
            if not self.points or not self.values:
                logger.error("Failed to extract valid points and values from quality data")
                return None
            
            # Get bounds
            self.bounds = self.get_bounds()

            # Create interpolation grid
            self.grid_x, self.grid_y, self.grid_z = self.create_grid()
            
            # Find global minimum
            results = self.run_all_algorithms()
            
            # Sort results by minimum value
            sorted_results = sorted(results, key=lambda x: x['minimum_value'])
            best_result = sorted_results[0]
            
            # Create visualization
            img_path = self.visualize_results(results)
            
            # Generate report
            report_file = os.path.join(
                self.report_output_dir, 
                "minima", 
                f"optimization_results_{timestamp}.txt"
            )
            
            with open(report_file, 'w') as f:
                f.write("=== OPTIMIZATION RESULTS ===\n")
                f.write(f"Input file: {self.json_file}\n")
                f.write(f"Data points: {len(self.points)}\n")
                f.write(f"Parameters: Teff range={self.bounds[0]}, log g range={self.bounds[1]}, z_scale={self.z_scale}\n")
                f.write("\n--- Global Minimum ---\n")
                f.write(f"Best algorithm: {best_result['algorithm']}\n")
                f.write(f"Minimum location: Teff={best_result['minimum_point'][0]:.2f}, log g={best_result['minimum_point'][1]:.5f}\n")
                f.write(f"Minimum Quality_shifted value: {best_result['minimum_value']:.8f}\n")
                f.write(f"\nVisualization saved to: {img_path}\n")
                f.write("\n--- All Results (sorted) ---\n")
                for i, result in enumerate(sorted_results):
                    f.write(f"{i+1}. {result['algorithm']}: "
                          f"Teff={result['minimum_point'][0]:.2f}, "
                          f"log g={result['minimum_point'][1]:.5f}, "
                          f"Quality={result['minimum_value']:.8f}")
                    if 'success' in result:
                        f.write(f" (Success: {result['success']})\n")
                    else:
                        f.write("\n")
            
                # Add information about DE and DA points
                f.write("\n--- Optimization Path Points ---\n")
                if self.de_points:
                    f.write(f"Differential Evolution: {len(self.de_points)} points\n")
                    de_result = next((r for r in results if r['algorithm'] == 'Differential Evolution'), None)
                    if de_result:
                        f.write(f"Final DE minimum: Teff={de_result['minimum_point'][0]:.2f}, "
                            f"log g={de_result['minimum_point'][1]:.5f}, "
                            f"Quality={de_result['minimum_value']:.8f}\n")
                
                if self.da_points:
                    f.write(f"Dual Annealing: {len(self.da_points)} points\n")
                    da_result = next((r for r in results if r['algorithm'] == 'Dual Annealing'), None)
                    if da_result:
                        f.write(f"Final DA minimum: Teff={da_result['minimum_point'][0]:.2f}, "
                            f"log g={da_result['minimum_point'][1]:.5f}, "
                            f"Quality={da_result['minimum_value']:.8f}\n")

            logger.info(f"Optimization results saved to {report_file}")
            
            # Also save results as JSON for use by other components
            json_results = {
                "best_result": {
                    "algorithm": best_result['algorithm'],
                    "minimum_point": best_result['minimum_point'].tolist(),
                    "minimum_value": float(best_result['minimum_value']),
                },
                "all_results": [{
                    "algorithm": r['algorithm'],
                    "minimum_point": r['minimum_point'].tolist(),
                    "minimum_value": float(r['minimum_value']),
                    "success": r.get('success', True)
                } for r in sorted_results],
                "optimization_paths": {
                    "differential_evolution": {
                        "points_count": len(self.de_points),
                        "sample_points": [p.tolist() for p in self.de_points[:5]] if len(self.de_points) > 0 else []
                    },
                    "dual_annealing": {
                        "points_count": len(self.da_points),
                        "sample_points": [p.tolist() for p in self.da_points[:5]] if len(self.da_points) > 0 else []
                    }
                }
            }
            
            json_report_file = os.path.join(
                self.report_output_dir, 
                "minima", 
                f"optimization_results_{timestamp}.json"
            )
            # Check if the best minimum is near an edge and suggest additional points if needed
            edge_points = self._check_minimum_near_edge(best_result, [p[0] for p in self.points], [p[1] for p in self.points])
            
            if edge_points:
                edge_report_file = os.path.join(
                    self.report_output_dir,
                    "minima",
                    f"edge_investigation_{timestamp}.txt"
                )
                
                with open(edge_report_file, 'w') as f:
                    f.write("=== EDGE INVESTIGATION RECOMMENDATIONS ===\n\n")
                    f.write("The global minimum is near the edge of the analyzed data range.\n")
                    f.write("To confirm if this is truly a global minimum, additional data points should be generated.\n\n")
                    f.write("Recommended additional points to generate:\n")
                    
                    for point in edge_points:
                        f.write(f"Teff={point[0]:.0f}, logg={point[1]:.2f}\n")
                    
                    f.write(f"\nTotal suggested points: {len(edge_points)}\n")
                
                # Add edge investigation to the JSON results
                json_results["edge_investigation"] = {
                    "is_near_edge": True,
                    "minimum_point": best_result['minimum_point'].tolist(),
                    "recommended_points": [point.tolist() if isinstance(point, np.ndarray) else point for point in edge_points]
                }
                
                logger.info(f"Edge investigation recommendations saved to {edge_report_file}")
            else:
                # Add edge investigation result to the JSON
                json_results["edge_investigation"] = {
                    "is_near_edge": False
                }
            
            # Save updated results as JSON
            with open(json_report_file, 'w') as f:
                json.dump(json_results, f, indent=4)
            
            return json_report_file
            
        except Exception as e:
            logger.error(f"Error in heatmap optimization: {e}")
            return None
    
    def load_data(self) -> Dict[str, Any]:
        """Load quality data from JSON file."""
        try:
            with open(self.json_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data from {self.json_file}: {e}")
            return {}
    
    def extract_points_values(self):
        """Extract points (teff, logg) and quality values from data."""
        self.points = []
        self.values = []
        excluded_points = []  # Track excluded points locally, not as instance variable
        
        # Create a filtered version of the data dictionary that will be used by other methods
        self.filtered_data = {}
        
        for key, item in self.data.items():
            try:
                teff = float(item['teff'])
                logg = float(item['logg'])
                quality = float(item['Quality_shifted'])
                
                # Store z_scale for reference
                if self.z_scale is None:
                    self.z_scale = item['z_scale']
                
                # Only include points where Teff is divisible by 100
                if teff % 100 == 0:
                    self.points.append([teff, logg])
                    self.values.append(quality)
                    self.filtered_data[key] = item  # Keep this item in the filtered data
                else:
                    # Store excluded points for reporting but don't add to self.points or self.values
                    excluded_points.append([teff, logg, quality])
                    logger.warning(f"Excluding point with Teff={teff}, not divisible by 100")
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping invalid data point {key}: {e}")
        
        if not self.points:
            logger.error("No valid data points extracted")
        
        # Save excluded points to a separate JSON file
        if excluded_points:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            special_points_dir = os.path.join(self.report_output_dir, "special_points")
            os.makedirs(special_points_dir, exist_ok=True)
            
            special_points_file = os.path.join(
                special_points_dir,
                f"special_points_{timestamp}.json"
            )
            
            special_points_data = {
                "special_points": [
                    {"teff": point[0], "logg": point[1], "quality": point[2]} 
                    for point in excluded_points
                ],
                "z_scale": self.z_scale
            }
            
            try:
                with open(special_points_file, 'w') as f:
                    json.dump(special_points_data, f, indent=4)
                logger.info(f"Saved {len(excluded_points)} special points to {special_points_file}")
            except Exception as e:
                logger.error(f"Failed to save special points: {e}")
                
        # Replace self.data with the filtered version to ensure excluded points don't affect other processing
        self.data = self.filtered_data

    def get_bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Get bounds for teff and logg values."""
        teffs = [p[0] for p in self.points]
        loggs = [p[1] for p in self.points]
        return ((min(teffs), max(teffs)), (min(loggs), max(loggs)))
    
    def create_grid(self, grid_density: int = 100, method: str = "linear") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a regular grid and interpolate the scattered data onto it.
        
        Args:
            grid_density: Number of grid points in each dimension
            method: Interpolation method for griddata ('cubic', 'linear', or 'nearest')
            
        Returns:
            grid_x, grid_y, grid_z: Mesh grid arrays for x, y and interpolated values
        """
        # Define grid bounds based on existing data
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        
        # Create a mesh grid
        grid_x, grid_y = np.mgrid[
            x_min:x_max:complex(0, grid_density), 
            y_min:y_max:complex(0, grid_density)
        ]
        
        # Use cubic interpolation as requested
        grid_z = griddata(self.points, self.values, (grid_x, grid_y), method=method)
        
        return grid_x, grid_y, grid_z

    def _check_minimum_near_edge(self, best_result: Dict[str, Any], x_values: List[float], y_values: List[float]) -> List[List[float]]:
        """
        Check if the global minimum is near an edge of the analyzed data range and
        suggest additional points to investigate if needed.
        
        Args:
            best_result: Dictionary with best optimization result
            x_values: List of existing Teff values
            y_values: List of existing logg values
            
        Returns:
            List of recommended points [Teff, logg] to generate, or empty list if not needed
        """
        # Only perform this check for global optimization methods
        if best_result['algorithm'] not in ['Differential Evolution', 'Dual Annealing']:
            return []
        
        minimum_point = best_result['minimum_point']
        min_teff, min_logg = minimum_point
        
        # Get data range
        min_teff_data, max_teff_data = min(x_values), max(x_values)
        min_logg_data, max_logg_data = min(y_values), max(y_values)
        
        # Check if minimum is near an edge (within 10% of range from edge)
        teff_range = max_teff_data - min_teff_data
        logg_range = max_logg_data - min_logg_data
        
        teff_buffer = 0.1 * teff_range
        logg_buffer = 0.1 * logg_range
        
        is_near_teff_min = min_teff < min_teff_data + teff_buffer
        is_near_teff_max = min_teff > max_teff_data - teff_buffer
        is_near_logg_min = min_logg < min_logg_data + logg_buffer
        is_near_logg_max = min_logg > max_logg_data - logg_buffer
        
        # If not near any edge, return empty list
        if not (is_near_teff_min or is_near_teff_max or is_near_logg_min or is_near_logg_max):
            return []
        
        logger.info(f"Global minimum at Teff={min_teff:.1f}, logg={min_logg:.4f} is near an edge of the data")
        
        # Create a set of existing points for easy lookup
        existing_points = {(round(x, 1), round(y, 2)) for x, y in zip(x_values, y_values)}
        
        # Define search range (±200 Teff, ±0.2 logg around minimum)
        teff_min_search = max(min_teff - 200, min_teff_data - 200)
        teff_max_search = min(min_teff + 200, max_teff_data + 200)
        logg_min_search = max(min_logg - 0.2, min_logg_data - 0.2)
        logg_max_search = min(min_logg + 0.2, max_logg_data + 0.2)
        
        # Generate recommended points with appropriate step sizes
        recommended_points = []
        
        # Generate Teff values in steps of 100
        teff_values = []
        current_teff = round(teff_min_search / 100) * 100  # Round to nearest 100
        while current_teff <= teff_max_search:
            teff_values.append(current_teff)
            current_teff += 100
        
        # Generate logg values in steps of 0.05
        logg_values = []
        current_logg = round(logg_min_search / 0.05) * 0.05  # Round to nearest 0.05
        while current_logg <= logg_max_search:
            logg_values.append(round(current_logg, 2))  # Ensure 2 decimal precision
            current_logg += 0.05
            current_logg = round(current_logg, 2)  # Avoid floating point issues
        
        # Create grid of points and filter existing ones
        for teff in teff_values:
            for logg in logg_values:
                point_key = (round(teff, 1), round(logg, 2))
                # Only add if point doesn't already exist
                if point_key not in existing_points:
                    recommended_points.append([teff, logg])
        
        logger.info(f"Recommended {len(recommended_points)} additional points to investigate edge minimum")
        return recommended_points

    def adjust_start_point(self, start_point: List[float], epsilon: float = 1e-3) -> List[float]:
        """
        Ensure the starting point is strictly inside the bounds by nudging it if needed.
        
        Args:
            start_point: Starting point [x, y]
            epsilon: Small value to nudge points away from bounds
            
        Returns:
            Adjusted start point
        """
        x, y = start_point
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        
        if x <= x_min:
            x = x_min + epsilon
        elif x >= x_max:
            x = x_max - epsilon
        if y <= y_min:
            y = y_min + epsilon
        elif y >= y_max:
            y = y_max - epsilon
        return [x, y]
    
    def objective_function(self, point: np.ndarray) -> float:
        """
        Interpolate the quality value at the given point with robust error handling.
        
        Args:
            point: Point to evaluate [x, y]
            
        Returns:
            Interpolated quality value, or inf if outside bounds or error
        """
        x, y = point
        
        # Check if point is within bounds (with a small buffer)
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        buffer = 0.001
        if (x < x_min + buffer or x > x_max - buffer or
            y < y_min + buffer or y > y_max - buffer):
            return float('inf')
        
        try:
            # Use safer linear interpolation first
            interpolated_value = griddata(self.points, self.values, ([x, y]), method='linear')
            if np.isnan(interpolated_value):
                interpolated_value = griddata(self.points, self.values, ([x, y]), method='cubic')
            if np.isnan(interpolated_value):
                return float('inf')
            return float(interpolated_value[0])
        except Exception as e:
            logger.warning(f"Error in interpolation at point {point}: {e}")
            return float('inf')
    
    def generate_random_start_point(self) -> List[float]:
        """Generate a random starting point within bounds."""
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        return [x, y]
    
    def find_minimum_nelder_mead(self, start_point: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Find minimum using Nelder-Mead algorithm.
        
        Args:
            start_point: Optional starting point, random if None
            
        Returns:
            Dictionary with optimization results
        """
        if start_point is None:
            start_point = self.generate_random_start_point()
        start_point = self.adjust_start_point(start_point)
        
        options = {'return_all': True, 'maxiter': 2000, 'adaptive': True}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            result = minimize(
                self.objective_function, 
                start_point, 
                method='Nelder-Mead',
                options=options
            )
        return {
            'algorithm': 'Nelder-Mead',
            'start_point': start_point,
            'minimum_point': result.x,
            'minimum_value': result.fun,
            'all_points': result.allvecs if hasattr(result, 'allvecs') else [],
            'success': result.success,
            'message': result.message
        }
    
    def find_minimum_powell(self, start_point: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Find minimum using Powell algorithm.
        
        Args:
            start_point: Optional starting point, random if None
            
        Returns:
            Dictionary with optimization results
        """
        if start_point is None:
            start_point = self.generate_random_start_point()
        start_point = self.adjust_start_point(start_point)
        
        options = {'return_all': True, 'maxiter': 2000}
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            result = minimize(
                self.objective_function, 
                start_point, 
                method='Powell',
                options=options
            )
        return {
            'algorithm': 'Powell',
            'start_point': start_point,
            'minimum_point': result.x,
            'minimum_value': result.fun,
            'all_points': result.allvecs if hasattr(result, 'allvecs') else [],
            'success': result.success,
            'message': result.message
        }
    
    def find_minimum_tnc(self, start_point: Optional[List[float]] = None) -> Dict[str, Any]:
        """
        Find minimum using TNC algorithm.
        
        Args:
            start_point: Optional starting point, random if None
            
        Returns:
            Dictionary with optimization results
        """
        if start_point is None:
            start_point = self.generate_random_start_point()
        start_point = self.adjust_start_point(start_point)
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            result = minimize(
                self.objective_function,
                start_point,
                method='TNC',
                bounds=self.bounds
            )
        return {
            'algorithm': 'TNC',
            'minimum_point': result.x,
            'minimum_value': result.fun,
            'success': result.success,
            'message': result.message
        }
    
    def de_callback(self, x: np.ndarray, convergence: float):
        """Callback for differential evolution to store intermediate solutions."""
        self.de_points.append(np.copy(x))
    
    def find_minimum_differential_evolution(self) -> Dict[str, Any]:
        """
        Find minimum using Differential Evolution (global optimizer).
        
        Returns:
            Dictionary with optimization results
        """
        self.de_points = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            result = differential_evolution(
                self.objective_function, 
                self.bounds,
                popsize=20,
                tol=0.01,
                callback=self.de_callback,
                polish=True
            )
        return {
            'algorithm': 'Differential Evolution',
            'minimum_point': result.x,
            'minimum_value': result.fun,
            'all_points': self.de_points,
            'success': result.success,
            'message': result.message
        }
    
    def da_callback(self, x: np.ndarray, f: float, context: int) -> bool:
        """Callback for dual annealing to store intermediate solutions."""
        self.da_points.append(np.copy(x))
        return False
    
    def find_minimum_dual_annealing(self) -> Dict[str, Any]:
        """
        Find minimum using Dual Annealing (global optimizer).
        
        Returns:
            Dictionary with optimization results
        """
        self.da_points = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            result = dual_annealing(
                self.objective_function, 
                self.bounds,
                callback=self.da_callback
            )
        return {
            'algorithm': 'Dual Annealing',
            'minimum_point': result.x,
            'minimum_value': result.fun,
            'all_points': self.da_points,
            'success': result.success,
            'message': result.message
        }
    
    def run_all_algorithms(self) -> List[Dict[str, Any]]:
        """
        Run all optimization algorithms in parallel with progress tracking.
        
        Returns:
            List of optimization results
        """
        results = []
        
        # 1. Run local methods in parallel
        logger.info("Running local optimization methods...")
        local_futures = []
        with ThreadPoolExecutor(max_workers=self.threads) as executor:
            for start_point in self.points:
                local_futures.append(executor.submit(self.find_minimum_nelder_mead, start_point))
                local_futures.append(executor.submit(self.find_minimum_powell, start_point))
                local_futures.append(executor.submit(self.find_minimum_tnc, start_point))
            
            for f in tqdm(as_completed(local_futures), total=len(local_futures), desc="Local Methods"):
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.error(f"Error in local optimization: {e}")
        
        # 2. Run global methods in parallel
        logger.info("Running global optimization methods...")
        global_futures = []
        with ThreadPoolExecutor(max_workers=min(2, self.threads)) as executor:
            global_futures.append(executor.submit(self.find_minimum_differential_evolution))
            global_futures.append(executor.submit(self.find_minimum_dual_annealing))
            
            for f in tqdm(as_completed(global_futures), total=len(global_futures), desc="Global Methods"):
                try:
                    results.append(f.result())
                except Exception as e:
                    logger.error(f"Error in global optimization: {e}")
        
        # Filter results to remove TNC results that duplicate input points
        filtered_results = []
        for r in results:
            if r['algorithm'] == 'TNC':
                # Check if this TNC result is essentially identical to one of the input points
                if any(np.allclose(np.array(p), r['minimum_point'], atol=1e-3) for p in self.points):
                    continue  # Skip this result
            filtered_results.append(r)
        
        return filtered_results
    
    def visualize_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Create a visualization of optimization results.
        
        Args:
            results: List of optimization results
            
        Returns:
            Path to the saved visualization image
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.image_output_dir, f"optimization_results_{timestamp}.png")
        
        plt.figure(figsize=(14, 10))
        
        # Create contour plot of the heatmap
        contour = plt.contourf(self.grid_x, self.grid_y, self.grid_z, 100, cmap='viridis')
        cbar = plt.colorbar(contour, label='Quality (Shifted)', pad=0.01)
        cbar.ax.tick_params(labelsize=10)
        
        # Plot original data points
        plt.scatter(
            [p[0] for p in self.points],
            [p[1] for p in self.points],
            color='white', marker='.', s=30, label='Data Points'
        )
        
        # Define markers for different algorithms
        markers = {
            'Nelder-Mead': 'o',
            'Powell': 's',
            'Differential Evolution': 'D',
            'Dual Annealing': 'P',
            'TNC': 'X',
        }
        
        # Get list of unique algorithms
        algo_list = list(set(r['algorithm'] for r in results))
        
        # Generate colors for algorithms
        colors = plt.cm.tab10(np.linspace(0, 1, len(algo_list)))
        color_map = {algo_list[i]: colors[i] for i in range(len(algo_list))}
        
        # Group results by algorithm
        algo_results = {}
        for result in results:
            algo = result['algorithm']
            if algo not in algo_results:
                algo_results[algo] = []
            algo_results[algo].append(result)
        
        # Plot results for each algorithm
        for algo, algo_results_list in algo_results.items():
            # Find best result for this algorithm
            best_result = min(algo_results_list, key=lambda x: x['minimum_value'])
            
            # Plot all results for this algorithm
            for i, r in enumerate(algo_results_list):
                label_str = None
                if i == 0:
                    label_str = f"{algo}: {best_result['minimum_value']:.8f}"
                
                plt.scatter(
                    r['minimum_point'][0],
                    r['minimum_point'][1],
                    color=color_map[algo],
                    marker=markers[algo],
                    s=70,
                    label=label_str
                )
            
            # Annotate best result with its value
            plt.annotate(
                f"{best_result['minimum_value']:.8f}",
                (best_result['minimum_point'][0], best_result['minimum_point'][1]),
                xytext=(5, 5), textcoords='offset points', fontsize=8
            )
        
        # Set plot properties
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.title(f'Heatmap of Quality (Shifted) with z_scale={self.z_scale}', fontsize=14)
        plt.xlabel('Effective Temperature (Teff)', fontsize=12)
        plt.ylabel('Surface Gravity (log g)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10, framealpha=0.9, markerscale=1.0, scatterpoints=1)
        
        # Save the figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualization saved to {output_path}")
        return output_path