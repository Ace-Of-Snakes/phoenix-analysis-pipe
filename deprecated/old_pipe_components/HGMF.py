import json
import numpy as np
import random
import argparse
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import basinhopping
from scipy.interpolate import griddata
from scipy.optimize import (
    minimize, 
    differential_evolution, 
    shgo, 
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
    
    def create_grid(self, grid_density=100):
        x_min, x_max = self.bounds[0]
        y_min, y_max = self.bounds[1]
        
        grid_x, grid_y = np.mgrid[
            x_min:x_max:complex(0, grid_density), 
            y_min:y_max:complex(0, grid_density)
        ]
        
        grid_z = griddata(self.points, self.values, (grid_x, grid_y), method='cubic')
        
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
        
        # Try to interpolate the value
        try:
            # Use safer linear interpolation first
            interpolated_value = griddata(self.points, self.values, ([x, y]), method='linear')
            
            # If linear gives NaN, try cubic
            if np.isnan(interpolated_value):
                interpolated_value = griddata(self.points, self.values, ([x, y]), method='cubic')
                
            # If still NaN, return infinity
            if np.isnan(interpolated_value):
                return float('inf')
                
            return float(interpolated_value[0])
        
        except Exception as e:
            # If any error occurs, return infinity
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
        
        options = {'return_all': True, 'maxiter': 1000, 'adaptive': True}
        
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
        # Adjust the starting point to be strictly inside the bounds
        start_point = self.adjust_start_point(start_point)
        
        options = {'return_all': True, 'maxiter': 1000}
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            # Remove bounds from the call to Powell to avoid internal errors.
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

    def find_minimum_basinhopping(self, start_point=None):
        """Use Basin-Hopping with L-BFGS-B as the local minimizer."""
        if start_point is None:
            start_point = self.generate_random_start_point()
        start_point = self.adjust_start_point(start_point)

        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': self.bounds
        }

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            result = basinhopping(
                self.objective_function,
                start_point,
                minimizer_kwargs=minimizer_kwargs,
                niter=100,  # increase if you want more thorough searching
                stepsize=50,  # size of random step
                T=1.0,       # "temperature" in Metropolis acceptance
            )

        return {
            'algorithm': 'Basin-Hopping',
            'minimum_point': result.x,
            'minimum_value': result.fun,
            'success': result.lowest_optimization_result.success,
            'message': result.message
        }
    
    def find_minimum_tnc(self, start_point=None):
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

    # def find_minimum_bfgs(self, start_point=None):
    #     """Find minimum using L-BFGS-B algorithm with bounds, adjusting the starting point if needed."""
    #     if start_point is None:
    #         start_point = self.generate_random_start_point()
    #     start_point = self.adjust_start_point(start_point)
        
    #     options = {'maxiter': 1000}
        
    #     with warnings.catch_warnings():
    #         warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
    #         result = minimize(
    #             self.objective_function, 
    #             start_point, 
    #             method='L-BFGS-B',
    #             bounds=self.bounds,
    #             options=options
    #         )
        
    #     return {
    #         'algorithm': 'L-BFGS-B',
    #         'start_point': start_point,
    #         'minimum_point': result.x,
    #         'minimum_value': result.fun,
    #         'all_points': [],
    #         'success': result.success,
    #         'message': result.message
    #     }

    
    def de_callback(self, x, convergence):
        """Callback for differential evolution to store intermediate solutions."""
        # x is the current best solution; store a copy
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
                polish=True  # local refinement
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
        # x is the current best solution; store a copy
        self.da_points.append(np.copy(x))
        # Return False to tell the optimizer to continue
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
    
    def find_minimum_shgo(self):
        """Find minimum using SHGO (global optimizer)."""
        # SHGO can return multiple local minima in result.xl
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'invalid value encountered', RuntimeWarning)
            result = shgo(
                self.objective_function, 
                self.bounds
            )
        
        return {
            'algorithm': 'SHGO',
            'minimum_point': result.x,
            'minimum_value': result.fun,
            'all_points': result.xl,  # local minima found by SHGO
            'success': result.success,
            'message': result.message
        }
        
    def run_all_algorithms(self):
        results = []
        
        # Use every given point from the data as a starting point
        for start_point in self.points:
            results.append(self.find_minimum_nelder_mead(start_point))
            results.append(self.find_minimum_powell(start_point))
            results.append(self.find_minimum_tnc(start_point))
            results.append(self.find_minimum_basinhopping(start_point))
            # results.append(self.find_minimum_bfgs(start_point))
        
        # Run global optimizers (they don't need starting points)
        results.append(self.find_minimum_differential_evolution())
        results.append(self.find_minimum_dual_annealing())
        results.append(self.find_minimum_shgo())
        
        return results


    
    def visualize_results(self, results):
        """Visualize the heatmap and optimization results with improved readability."""
        plt.figure(figsize=(14, 10))
        
        # Plot the heatmap
        contour = plt.contourf(self.grid_x, self.grid_y, self.grid_z, 100, cmap='viridis')
        cbar = plt.colorbar(contour, label='Quality (Shifted)', pad=0.01)
        cbar.ax.tick_params(labelsize=10)
        
        # Plot the original data points
        plt.scatter(
            [p[0] for p in self.points],
            [p[1] for p in self.points],
            color='white', marker='.', s=30, label='Data Points'
        )
        
        # Markers for different algorithms
        markers = {
            'Nelder-Mead': 'o',
            'Powell': 's',
            'Differential Evolution': 'D',
            'Dual Annealing': 'P',
            'SHGO': '*',
            'TNC': 'X',
            'Basin-Hopping': 'H',
        }
        
        # Distinct colors for each algorithm
        algo_list = list(set(r['algorithm'] for r in results))
        colors = plt.cm.tab10(np.linspace(0, 1, len(algo_list)))
        color_map = {algo_list[i]: colors[i] for i in range(len(algo_list))}
        
        # Group results by algorithm
        algo_results = {}
        for result in results:
            algo = result['algorithm']
            if algo not in algo_results:
                algo_results[algo] = []
            algo_results[algo].append(result)
        
        # Plot all final results from each algorithm
        for algo, algo_results_list in algo_results.items():
            # Identify the best result (lowest Quality) for annotation
            best_result = min(algo_results_list, key=lambda x: x['minimum_value'])
            
            # Plot *every* final point from this algorithm
            for i, r in enumerate(algo_results_list):
                # Give the algorithm label only to the *first* point so legend is not repeated
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
            
            # Annotate only the best result with its Quality value
            plt.annotate(
                f"{best_result['minimum_value']:.8f}",
                (best_result['minimum_point'][0], best_result['minimum_point'][1]),
                xytext=(5, 5), textcoords='offset points', fontsize=8
            )
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # Add title and labels
        plt.title(f'Heatmap of Quality (Shifted) with z_scale={self.z_scale}', fontsize=14)
        plt.xlabel('Effective Temperature (Teff)', fontsize=12)
        plt.ylabel('Surface Gravity (log g)', fontsize=12)
        
        # More readable legend
        plt.legend(loc='upper right', fontsize=10, framealpha=0.9, markerscale=1.0, scatterpoints=1)
        
        plt.tight_layout()
        output_path = 'optimization_results.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return output_path

    
    def find_global_minimum(self):
        """Find the global minimum using multiple algorithms and starting points."""
        results = self.run_all_algorithms()
        
        # Sort results by minimum value
        sorted_results = sorted(results, key=lambda x: x['minimum_value'])
        
        # Get the best result overall
        best_result = sorted_results[0]
        
        # Visualize results
        img_path = self.visualize_results(results)
        
        # Print a detailed summary
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
