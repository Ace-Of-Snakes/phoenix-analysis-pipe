import json
import numpy as np
import random
import argparse
import warnings
import matplotlib.pyplot as plt

# NEW imports for progress bars & parallel execution
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from scipy.interpolate import griddata
from scipy.optimize import (
    minimize, 
    differential_evolution,
    dual_annealing
)

class HeatmapOptimizer:
    def __init__(self, json_file):
        self.json_file = json_file
        self.data = self.load_data()
        self.points = []
        self.values = []
        self.z_scale = None
        self.extract_points_values()
        self.bounds = self.get_bounds()
        self.grid_x, self.grid_y, self.grid_z = self.create_grid()
        
        # Lists to store intermediate solutions from global optimizers
        self.de_points = []
        self.da_points = []
    
    def adjust_start_point(self, start_point, epsilon=1e-3):
        """Ensure the starting point is strictly inside the bounds by nudging it if needed."""
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
    
    def load_data(self):
        with open(self.json_file, 'r') as f:
            return json.load(f)
    
    def extract_points_values(self):
        for key, item in self.data.items():
            teff = float(item['teff'])
            logg = float(item['logg'])
            quality = float(item['Quality_shifted'])
            
            # Store z_scale for reference
            if self.z_scale is None:
                self.z_scale = item['z_scale']
            
            self.points.append([teff, logg])
            self.values.append(quality)
    
    def get_bounds(self):
        teffs = [p[0] for p in self.points]
        loggs = [p[1] for p in self.points]
        return [(min(teffs), max(teffs)), (min(loggs), max(loggs))]
    
    def create_grid(self, grid_density=100, interp_method="griddata", griddata_method="cubic"):
        """
        Create a regular grid and interpolate the scattered data onto it.

        Parameters:
            grid_density (int): Number of grid points in each dimension.
            interp_method (str): Interpolation method to use. Options:
                                - "griddata": Uses scipy.interpolate.griddata.
                                - "rbf": Uses scipy.interpolate.Rbf.
                                - "spline": Uses scipy.interpolate.SmoothBivariateSpline.
            griddata_method (str): When using "griddata", the interpolation kind ("cubic", "linear", or "nearest").

        Returns:
            grid_x, grid_y, grid_z: The meshgrid arrays for x and y and the interpolated values.
        """
        # Define grid bounds based on the existing data
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        
        # Create a mesh grid using mgrid
        grid_x, grid_y = np.mgrid[
            x_min:x_max:complex(0, grid_density), 
            y_min:y_max:complex(0, grid_density)
        ]
        
        if interp_method == "griddata":
            # Standard griddata interpolation with the chosen method
            grid_z = griddata(self.points, self.values, (grid_x, grid_y), method=griddata_method)
        
        elif interp_method == "rbf":
            # Use Radial Basis Functions for interpolation
            from scipy.interpolate import Rbf
            # Convert list of points to a NumPy array and separate coordinates
            points_arr = np.array(self.points)
            x_data = points_arr[:, 0]
            y_data = points_arr[:, 1]
            values = np.array(self.values)
            # Create an RBF interpolator; you can change 'function' to another type (e.g., 'linear', 'gaussian')
            rbf_interpolator = Rbf(x_data, y_data, values, function='multiquadric')
            grid_z = rbf_interpolator(grid_x, grid_y)
        
        elif interp_method == "spline":
            # Use SmoothBivariateSpline for a spline-based interpolation
            from scipy.interpolate import SmoothBivariateSpline
            points_arr = np.array(self.points)
            x_data = points_arr[:, 0]
            y_data = points_arr[:, 1]
            values = np.array(self.values)
            # s=0 enforces interpolation through the points; adjust s for smoothing if needed.
            spline = SmoothBivariateSpline(x_data, y_data, values, s=0)
            # Create a new 1D grid for each dimension
            x_lin = np.linspace(x_min, x_max, grid_density)
            y_lin = np.linspace(y_min, y_max, grid_density)
            # Evaluate the spline on the 1D grid arrays; this returns a 2D array.
            grid_z = spline(x_lin, y_lin)
            # Rebuild grid_x and grid_y with meshgrid for consistency in visualization.
            grid_x, grid_y = np.meshgrid(x_lin, y_lin)
        
        else:
            raise ValueError(f"Unknown interpolation method: {interp_method}")

        return grid_x, grid_y, grid_z

    
    def objective_function(self, point):
        """Interpolates the quality value at the given point with robust error handling."""
        x, y = point
        
        # Check if point is within bounds (with a small buffer to avoid edge artifacts).
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
            print(f"Warning: Error in interpolation at point {point}: {e}")
            return float('inf')
    
    def generate_random_start_point(self):
        """Generate a random starting point within bounds."""
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        x = random.uniform(x_min, x_max)
        y = random.uniform(y_min, y_max)
        return [x, y]
    
    def find_minimum_nelder_mead(self, start_point=None):
        """Find minimum using Nelder-Mead algorithm, adjusting starting point if needed."""
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

    def find_minimum_powell(self, start_point=None):
        """Find minimum using Powell algorithm, adjusting starting point if needed."""
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
    
    def find_minimum_tnc(self, start_point=None):
        """Find minimum using TNC. TNC results that duplicate an input point will be filtered later."""
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
    
    def de_callback(self, x, convergence):
        """Callback for differential evolution to store intermediate solutions."""
        self.de_points.append(np.copy(x))
    
    def find_minimum_differential_evolution(self):
        """Find minimum using Differential Evolution (global optimizer)."""
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
    
    def da_callback(self, x, f, context):
        """Callback for dual annealing to store intermediate solutions."""
        self.da_points.append(np.copy(x))
        return False
    
    def find_minimum_dual_annealing(self):
        """Find minimum using Dual Annealing (global optimizer)."""
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

    def run_all_algorithms(self):
        """
        Run local methods in parallel with ThreadPoolExecutor and track progress with tqdm.
        Then run global optimizers.
        """
        results = []

        # --- 1) Parallel Execution of Local Methods ---
        local_futures = []
        with ThreadPoolExecutor() as executor:
            for start_point in self.points:
                local_futures.append(executor.submit(self.find_minimum_nelder_mead, start_point))
                local_futures.append(executor.submit(self.find_minimum_powell, start_point))
                local_futures.append(executor.submit(self.find_minimum_tnc, start_point))
            for f in tqdm(as_completed(local_futures), total=len(local_futures), desc="Local Methods"):
                results.append(f.result())

        # --- 2) Global Methods ---
        global_futures = []
        with ThreadPoolExecutor() as executor:
            global_futures.append(executor.submit(self.find_minimum_differential_evolution))
            global_futures.append(executor.submit(self.find_minimum_dual_annealing))
            for f in tqdm(as_completed(global_futures), total=len(global_futures), desc="Global Methods"):
                results.append(f.result())

        return results
    
    def visualize_results(self, results):
        """Visualize the heatmap and optimization results with improved readability."""
        plt.figure(figsize=(14, 10))
        contour = plt.contourf(self.grid_x, self.grid_y, self.grid_z, 100, cmap='viridis')
        cbar = plt.colorbar(contour, label='Quality (Shifted)', pad=0.01)
        cbar.ax.tick_params(labelsize=10)
        plt.scatter(
            [p[0] for p in self.points],
            [p[1] for p in self.points],
            color='white', marker='.', s=30, label='Data Points'
        )
        markers = {
            'Nelder-Mead': 'o',
            'Powell': 's',
            'Differential Evolution': 'D',
            'Dual Annealing': 'P',
            'TNC': 'X',
            'Basin-Hopping': 'H',
        }
        algo_list = list(set(r['algorithm'] for r in results))
        colors = plt.cm.tab10(np.linspace(0, 1, len(algo_list)))
        color_map = {algo_list[i]: colors[i] for i in range(len(algo_list))}
        algo_results = {}
        for result in results:
            algo = result['algorithm']
            if algo not in algo_results:
                algo_results[algo] = []
            algo_results[algo].append(result)
        for algo, algo_results_list in algo_results.items():
            best_result = min(algo_results_list, key=lambda x: x['minimum_value'])
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
            plt.annotate(
                f"{best_result['minimum_value']:.8f}",
                (best_result['minimum_point'][0], best_result['minimum_point'][1]),
                xytext=(5, 5), textcoords='offset points', fontsize=8
            )
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
        plt.title(f'Heatmap of Quality (Shifted) with z_scale={self.z_scale}', fontsize=14)
        plt.xlabel('Effective Temperature (Teff)', fontsize=12)
        plt.ylabel('Surface Gravity (log g)', fontsize=12)
        plt.legend(loc='upper right', fontsize=10, framealpha=0.9, markerscale=1.0, scatterpoints=1)
        plt.tight_layout()
        output_path = 'optimization_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return output_path

    def find_global_minimum(self):
        """Find the global minimum using multiple algorithms and starting points."""
        results = self.run_all_algorithms()
        
        # --- Filtering Step: Remove TNC results whose minimum_point is already in the input data ---
        filtered_results = []
        for r in results:
            if r['algorithm'] == 'TNC':
                # Check if this TNC result is essentially identical to one of the input points.
                if any(np.allclose(np.array(p), r['minimum_point'], atol=1e-3) for p in self.points):
                    continue  # Skip this result.
            filtered_results.append(r)
        results = filtered_results

        sorted_results = sorted(results, key=lambda x: x['minimum_value'])
        best_result = sorted_results[0]
        img_path = self.visualize_results(results)
        
        print("\n=== OPTIMIZATION RESULTS ===")
        print(f"Input file: {self.json_file}")
        print(f"Data points: {len(self.points)}")
        print(f"Parameters: Teff range={self.bounds[0]}, log g range={self.bounds[1]}, z_scale={self.z_scale}")
        print("\n--- Global Minimum ---")
        print(f"Best algorithm: {best_result['algorithm']}")
        print(f"Minimum location: Teff={best_result['minimum_point'][0]:.2f}, log g={best_result['minimum_point'][1]:.5f}")
        print(f"Minimum Quality_shifted value: {best_result['minimum_value']:.8f}")
        print(f"\nVisualization saved to: {img_path}")
        print("\n--- All Results (sorted) ---")
        for i, result in enumerate(sorted_results):
            print(f"{i+1}. {result['algorithm']}: "
                  f"Teff={result['minimum_point'][0]:.2f}, "
                  f"log g={result['minimum_point'][1]:.5f}, "
                  f"Quality={result['minimum_value']:.8f}")
        return best_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find global minimum in heatmap data')
    parser.add_argument('json_file', type=str, help='Path to the JSON file containing heatmap data')
    args = parser.parse_args()
    
    optimizer = HeatmapOptimizer(args.json_file)
    best_result = optimizer.find_global_minimum()
