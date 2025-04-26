#!/usr/bin/env python3
# report_generator.py

import os
import json
import numpy as np
import logging
import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Union

logger = logging.getLogger("ReportGenerator")

class ReportGenerator:
    """
    Generate comprehensive reports from pipeline results.
    """
    
    def __init__(self, quality_output_dir: str, image_output_dir: str, report_output_dir: str):
        """
        Initialize the report generator.
        
        Args:
            quality_output_dir: Directory containing quality calculation results
            image_output_dir: Directory for storing output images
            report_output_dir: Directory for storing reports
        """
        self.quality_output_dir = quality_output_dir
        self.image_output_dir = image_output_dir
        self.report_output_dir = report_output_dir
        
        # Ensure output directory exists
        self.summary_dir = os.path.join(self.report_output_dir, "summary")
        os.makedirs(self.summary_dir, exist_ok=True)
    
    def run(self, quality_results: Optional[str], minimum_results: Optional[str], 
           periodicity_results: Optional[str], z_scale_results: Optional[str],
           timestamp: Optional[str] = None) -> bool:
        """
        Generate a comprehensive report from pipeline results.
        
        Args:
            quality_results: Path to quality results file
            minimum_results: Path to minimum analysis results file
            periodicity_results: Path to periodicity analysis results file
            z_scale_results: Path to z-scale analysis results file
            timestamp: Optional timestamp for file naming
            
        Returns:
            True if successful, False otherwise
        """
        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Define output files
            summary_report = os.path.join(self.summary_dir, f"summary_report_{timestamp}.txt")
            summary_json = os.path.join(self.summary_dir, f"summary_report_{timestamp}.json")
            
            # Load all available results
            results_data = {
                'quality': self._load_json(quality_results) if quality_results else None,
                'minimum': self._load_json(minimum_results) if minimum_results else None,
                'periodicity': self._load_json(periodicity_results) if periodicity_results else None,
                'z_scale': self._load_json(z_scale_results) if z_scale_results else None
            }
            
            # Generate summary report text
            with open(summary_report, 'w', encoding='utf-8') as f:
                f.write(f"PHOENIX STELLAR ATMOSPHERE MODEL ANALYSIS\n")
                f.write(f"=======================================\n\n")
                f.write(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Quality Results Summary
                f.write(f"QUALITY CALCULATION SUMMARY\n")
                f.write(f"--------------------------\n")
                if results_data['quality']:
                    n_models = len(results_data['quality'])
                    
                    # Extract quality values
                    qualities = [float(data.get('Quality_shifted', 0)) for data in results_data['quality'].values()]
                    
                    f.write(f"Analyzed {n_models} stellar atmosphere models\n")
                    f.write(f"Quality statistics:\n")
                    f.write(f"  Mean: {np.mean(qualities):.8f}\n")
                    f.write(f"  Min: {np.min(qualities):.8f}\n")
                    f.write(f"  Max: {np.max(qualities):.8f}\n")
                    f.write(f"  Std Dev: {np.std(qualities):.8f}\n")
                else:
                    f.write(f"No quality calculation results available.\n")
                f.write("\n")
                
                # Global Minimum Summary
                f.write(f"GLOBAL MINIMUM ANALYSIS\n")
                f.write(f"----------------------\n")
                if results_data['minimum'] and 'best_result' in results_data['minimum']:
                    best = results_data['minimum']['best_result']
                    f.write(f"Best algorithm: {best.get('algorithm', 'Unknown')}\n")
                    
                    if 'minimum_point' in best:
                        min_point = best['minimum_point']
                        f.write(f"Minimum location: Teff={min_point[0]:.2f}, logg={min_point[1]:.5f}\n")
                    
                    f.write(f"Minimum quality value: {best.get('minimum_value', 'Unknown'):.8f}\n")
                    
                    if 'all_results' in results_data['minimum']:
                        f.write(f"\nAll optimization results:\n")
                        for i, result in enumerate(results_data['minimum']['all_results'][:5]):  # Top 5
                            f.write(f"  {i+1}. {result.get('algorithm', 'Unknown')}: "
                                  f"Quality={result.get('minimum_value', 'Unknown'):.8f}\n")
                        
                        if len(results_data['minimum']['all_results']) > 5:
                            f.write(f"  ... and {len(results_data['minimum']['all_results']) - 5} more\n")
                else:
                    f.write(f"No global minimum analysis results available.\n")
                f.write("\n")
                
                # Periodicity Analysis Summary
                f.write(f"PERIODICITY ANALYSIS SUMMARY\n")
                f.write(f"---------------------------\n")
                if results_data['periodicity']:
                    f.write(f"Detected periodicities:\n")
                    f.write(f"  Teff: {results_data['periodicity'].get('teff_periodicity', 'Unknown'):.2f}\n")
                    f.write(f"  logg: {results_data['periodicity'].get('logg_periodicity', 'Unknown'):.4f}\n")
                    
                    n_clusters = results_data['periodicity'].get('n_clusters', 0)
                    f.write(f"Identified {n_clusters} pattern clusters\n")
                    
                    pred_points = results_data['periodicity'].get('predicted_points', {})
                    if pred_points and 'teff' in pred_points:
                        n_predictions = len(pred_points['teff'])
                        f.write(f"Generated {n_predictions} suggestions for additional data points\n")
                else:
                    f.write(f"No periodicity analysis results available.\n")
                f.write("\n")
                
                # Z-scale Analysis Summary
                f.write(f"Z-SCALE ANALYSIS SUMMARY\n")
                f.write(f"-----------------------\n")
                if results_data['z_scale']:
                    # Extract z-scale values from all teff/logg pairs
                    z_scale_values = []
                    for pair_data in results_data['z_scale'].values():
                        if 'min_z_scale' in pair_data:
                            z_scale_values.append(float(pair_data['min_z_scale']))
                    
                    if z_scale_values:
                        mean_z_scale = np.mean(z_scale_values)
                        std_z_scale = np.std(z_scale_values)
                        
                        f.write(f"Analyzed {len(z_scale_values)} teff/logg pairs with multiple z-scales\n")
                        f.write(f"Mean optimal z-scale: {mean_z_scale:.4f} ± {std_z_scale:.4f}\n")
                        
                        if std_z_scale < 0.1:
                            f.write(f"Recommended z-scale for future models: {mean_z_scale:.4f}\n")
                        else:
                            f.write(f"Z-scale varies significantly across models; consider different z-scales\n")
                            f.write(f"for different temperature/gravity regimes. See detailed report.\n")
                else:
                    f.write(f"No z-scale analysis results available.\n")
                f.write("\n")
                
                # OVERALL RECOMMENDATIONS
                f.write(f"OVERALL RECOMMENDATIONS\n")
                f.write(f"======================\n\n")
                
                # Combine insights from all analyses
                min_point = None
                if results_data['minimum'] and 'best_result' in results_data['minimum']:
                    min_point = results_data['minimum']['best_result'].get('minimum_point')
                
                f.write(f"Based on the combined analysis, we recommend:\n\n")
                
                if min_point:
                    f.write(f"1. QUALITY OPTIMUM:\n")
                    f.write(f"   The optimal model parameters are approximately:\n")
                    f.write(f"   Teff = {min_point[0]:.1f}\n")
                    f.write(f"   logg = {min_point[1]:.4f}\n")
                
                if results_data['periodicity']:
                    teff_period = results_data['periodicity'].get('teff_periodicity')
                    logg_period = results_data['periodicity'].get('logg_periodicity')
                    
                    if teff_period and logg_period:
                        f.write(f"\n2. SAMPLING STRATEGY:\n")
                        f.write(f"   Sample additional models using these intervals:\n")
                        f.write(f"   Teff: Sample every {teff_period:.1f} K\n")
                        f.write(f"   logg: Sample every {logg_period:.4f}\n")
                
                if results_data['z_scale'] and 'z_scale_values' in locals() and z_scale_values:
                    f.write(f"\n3. Z-SCALE RECOMMENDATION:\n")
                    if std_z_scale < 0.1:
                        f.write(f"   Use a consistent z-scale of {mean_z_scale:.4f} across all models\n")
                    else:
                        f.write(f"   Z-scale varies across temperature/gravity regimes.\n")
                        f.write(f"   See detailed z-scale report for specific recommendations.\n")
                
                # Final note
                f.write(f"\nFull details and visualizations are available in the respective analysis reports.\n")
            
            # Generate summary JSON
            self.summary_json_data = {
                'timestamp': timestamp,
                'quality_summary': None,
                'minimum_summary': None,
                'periodicity_summary': None,
                'z_scale_summary': None
            }
            
            # Quality summary
            if results_data['quality']:
                qualities = [float(data.get('Quality_shifted', 0)) for data in results_data['quality'].values()]
                self.summary_json_data['quality_summary'] = {
                    'n_models': len(results_data['quality']),
                    'mean_quality': float(np.mean(qualities)),
                    'min_quality': float(np.min(qualities)),
                    'max_quality': float(np.max(qualities)),
                    'std_quality': float(np.std(qualities))
                }
            
            # Minimum summary
            if results_data['minimum'] and 'best_result' in results_data['minimum']:
                best = results_data['minimum']['best_result']
                self.summary_json_data['minimum_summary'] = {
                    'algorithm': best.get('algorithm'),
                    'minimum_point': best.get('minimum_point'),
                    'minimum_value': best.get('minimum_value')
                }
                
                # Include edge investigation if available
                if 'edge_investigation' in results_data['minimum']:
                    edge_info = results_data['minimum']['edge_investigation']
                    self.summary_json_data['minimum_summary']['is_near_edge'] = edge_info.get('is_near_edge', False)
                    
                    if edge_info.get('is_near_edge', False) and 'recommended_points' in edge_info:
                        self.summary_json_data['minimum_summary']['recommended_points'] = edge_info['recommended_points']
                        
                        # Add edge investigation info to the report
                        f.write("\n4. EDGE INVESTIGATION:\n")
                        f.write("   The global minimum is near the edge of the analyzed data range.\n")
                        f.write("   Additional data points are recommended to confirm if this is truly a global minimum.\n")
                        f.write(f"   See detailed edge investigation report for {len(edge_info['recommended_points'])} recommended points.\n")
            
            # Periodicity summary
            if results_data['periodicity']:
                self.summary_json_data['periodicity_summary'] = {
                    'teff_periodicity': results_data['periodicity'].get('teff_periodicity'),
                    'logg_periodicity': results_data['periodicity'].get('logg_periodicity'),
                    'n_clusters': results_data['periodicity'].get('n_clusters'),
                    'n_predictions': len(results_data['periodicity'].get('predicted_points', {}).get('teff', []))
                }
            
            # Z-scale summary
            if results_data['z_scale'] and 'z_scale_values' in locals() and z_scale_values:
                self.summary_json_data['z_scale_summary'] = {
                    'n_pairs': len(z_scale_values),
                    'mean_z_scale': float(mean_z_scale),
                    'std_z_scale': float(std_z_scale),
                    'recommended_z_scale': float(mean_z_scale) if std_z_scale < 0.1 else None
                }
            
            # Write summary JSON
            with open(summary_json, 'w', encoding='utf-8') as f:
                json.dump(self.summary_json_data, f, indent=4)
            
            # Create a combined visualization if all components are available
            if (results_data['quality'] and results_data['minimum'] and 
                results_data['periodicity'] and min_point and 
                'predicted_points' in results_data['periodicity']):
                
                summary_image = os.path.join(self.image_output_dir, f"summary_{timestamp}.png")
                self._create_summary_visualization(
                    results_data, min_point, summary_image
                )
            
            logger.info(f"Generated summary report: {summary_report}")
            return True
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return False
    
    def _load_json(self, file_path: str) -> Dict:
        """Load data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load data from {file_path}: {e}")
            return {}
    
    def _create_summary_visualization(self, results_data: Dict, min_point: List[float], output_file: str):
        """
        Create a summary visualization combining key insights from all analyses.
        
        Args:
            results_data: Dictionary with all results data
            min_point: Global minimum point [teff, logg]
            output_file: Path to save the visualization
        """
        try:
            # Extract original data points from quality results
            teff_values = []
            logg_values = []
            quality_values = []
            
            for filename, data in results_data['quality'].items():
                teff_values.append(float(data.get('teff', 0)))
                logg_values.append(float(data.get('logg', 0)))
                quality_values.append(float(data.get('Quality_shifted', 0)))
            
            teff_values = np.array(teff_values)
            logg_values = np.array(logg_values)
            quality_values = np.array(quality_values)
            
            # Extract predicted points
            pred_points = results_data['periodicity'].get('predicted_points', {})
            if pred_points and 'teff' in pred_points and 'logg' in pred_points:
                pred_teff = np.array(pred_points['teff'])
                pred_logg = np.array(pred_points['logg'])
            else:
                pred_teff = np.array([])
                pred_logg = np.array([])
            
            # Create the visualization
            plt.figure(figsize=(12, 10))
            
            # Create a contour plot of the quality values
            if len(teff_values) >= 4:  # Need at least a few points for interpolation
                # Create a grid for the contour plot
                xi = np.linspace(min(teff_values), max(teff_values), 100)
                yi = np.linspace(min(logg_values), max(logg_values), 100)
                
                try:
                    from scipy.interpolate import griddata
                    zi = griddata((teff_values, logg_values), quality_values, 
                                (xi[None,:], yi[:,None]), method='cubic')
                    
                    # Handle NaN values
                    if np.any(np.isnan(zi)):
                        zi = np.nan_to_num(zi, nan=np.nanmean(quality_values))
                    
                    # Plot the contour
                    contour = plt.contourf(xi, yi, zi, 100, cmap='viridis')
                    plt.colorbar(contour, label='Quality (Shifted)')
                except Exception as e:
                    logger.warning(f"Contour plot failed: {e}")
                    # Fallback to scatter plot with color mapping
                    scatter = plt.scatter(teff_values, logg_values, c=quality_values, 
                                       cmap='viridis', s=80, alpha=0.7)
                    plt.colorbar(scatter, label='Quality (Shifted)')
            else:
                # With very few points, just use scatter plot
                scatter = plt.scatter(teff_values, logg_values, c=quality_values, 
                                   cmap='viridis', s=80, alpha=0.7)
                plt.colorbar(scatter, label='Quality (Shifted)')
            
            # Plot original data points
            plt.scatter(teff_values, logg_values, marker='o', s=60, edgecolor='white', 
                      color='black', alpha=0.5, label='Original Data')
            
            # Plot predicted points
            if len(pred_teff) > 0:
                # Take a random sample if there are many points
                max_display = 50
                if len(pred_teff) > max_display:
                    indices = np.random.choice(len(pred_teff), max_display, replace=False)
                    plt.scatter(pred_teff[indices], pred_logg[indices], marker='x', s=50, 
                             color='red', label=f'Predicted Points (sample of {max_display})')
                else:
                    plt.scatter(pred_teff, pred_logg, marker='x', s=50, 
                             color='red', label='Predicted Points')
            
            # Highlight the global minimum
            plt.scatter([min_point[0]], [min_point[1]], marker='*', s=300, 
                      color='yellow', edgecolor='black', linewidth=1.5,
                      label=f'Global Minimum: Teff={min_point[0]:.1f}, logg={min_point[1]:.4f}')
            
            # If we have periodicity information, show it
            if 'teff_periodicity' in results_data['periodicity'] and 'logg_periodicity' in results_data['periodicity']:
                teff_period = results_data['periodicity']['teff_periodicity']
                logg_period = results_data['periodicity']['logg_periodicity']
                
                # Center point for periodicity grid
                teff_center = np.mean([min(teff_values), max(teff_values)])
                logg_center = np.mean([min(logg_values), max(logg_values)])
                
                # Draw periodicity grid lines
                for i in range(-5, 6):
                    teff_line = teff_center + i * teff_period
                    if min(teff_values) <= teff_line <= max(teff_values):
                        plt.axvline(teff_line, color='white', linestyle='--', alpha=0.3)
                
                for i in range(-5, 6):
                    logg_line = logg_center + i * logg_period
                    if min(logg_values) <= logg_line <= max(logg_values):
                        plt.axhline(logg_line, color='white', linestyle='--', alpha=0.3)
                
                # Add periodicity information to the title
                period_info = f"Periodicities: Teff={teff_period:.1f}, logg={logg_period:.4f}"
            else:
                period_info = ""
            
            # Add z-scale information if available
            if results_data['z_scale'] and 'z_scale_summary' in locals():
                z_scale_info = f"Optimal z-scale: {self.summary_json_data['z_scale_summary']['mean_z_scale']:.4f}"
            else:
                z_scale_info = ""
            
            # Set labels, title, and grid
            plt.xlabel('Effective Temperature (Teff)', fontsize=12)
            plt.ylabel('Surface Gravity (logg)', fontsize=12)
            
            title = "PHOENIX Model Analysis Summary"
            if period_info:
                title += f"\n{period_info}"
            if z_scale_info:
                title += f"\n{z_scale_info}"
                
            plt.title(title, fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.3)
            plt.legend(loc='best', fontsize=10)
            
            # Save the figure
            plt.tight_layout()
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Summary visualization saved to {output_file}")
            
        except Exception as e:
            logger.error(f"Failed to create summary visualization: {e}")