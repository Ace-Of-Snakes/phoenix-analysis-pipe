#!/usr/bin/env python3
# phoenix_analysis_pipe.py

import os
import sys
import json
import logging
import argparse
import datetime
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import pipeline components
from quality_calculator import QualityCalculator
from heatmap_optimizer import HeatmapMinimumFinder
from periodicity_detector import PeriodicityDetector
from neural_periodicity_detector import NeuralPeriodicityDetector
from z_scale_optimizer import ZScaleOptimizer
from report_generator import ReportGenerator

# Setup logging
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

@dataclass
class PipelineConfig:
    """Configuration for the PHOENIX model analysis pipeline."""
    nlte_models_dir: str
    z_scale_models_dir: str
    quality_output_dir: str
    image_output_dir: str
    report_output_dir: str
    recalculate_all: bool = False
    skip_z_scale_min: bool = False
    skip_prediction: bool = False
    skip_neural_prediction: bool = False
    config_file: Optional[str] = None
    original_spectrum_file: str = "uves_spectra_fomalhaut.csv"
    threads: int = max(1, os.cpu_count() - 1)
    log_level: str = "INFO"

class PhoenixPipeline:
    """Main pipeline for PHOENIX model analysis."""
    
    def __init__(self, config: PipelineConfig):
        self.config = config

        # Setup logging
        self.log_dir = os.path.join(os.getcwd(), config.report_output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger("PhoenixPipeline")
        self.logger.setLevel(getattr(logging, config.log_level))
        
        # Create file handler
        log_file = os.path.join(os.getcwd(), self.log_dir, f"pipeline_{timestamp}.log")
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(file_handler)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(log_format))
        self.logger.addHandler(console_handler)
        
        # Initialize pipeline components
        self.quality_calculator = QualityCalculator(
            nlte_models_dir=config.nlte_models_dir,
            quality_output_dir=config.quality_output_dir,
            original_spectrum_file=config.original_spectrum_file,
            recalculate_all=config.recalculate_all,
            threads=config.threads
        )
        
        self.heatmap_optimizer = HeatmapMinimumFinder(
            quality_output_dir=config.quality_output_dir,
            image_output_dir=config.image_output_dir,
            report_output_dir=config.report_output_dir,
            threads=config.threads
        )
        
        self.periodicity_detector = PeriodicityDetector(
            quality_output_dir=config.quality_output_dir,
            image_output_dir=config.image_output_dir,
            report_output_dir=config.report_output_dir
        )
        
        self.neural_periodicity_detector = NeuralPeriodicityDetector(
            quality_output_dir=config.quality_output_dir,
            image_output_dir=config.image_output_dir,
            report_output_dir=config.report_output_dir
        )
        
        self.z_scale_optimizer = ZScaleOptimizer(
            z_scale_models_dir=config.z_scale_models_dir,
            quality_output_dir=config.quality_output_dir,
            image_output_dir=config.image_output_dir,
            report_output_dir=config.report_output_dir,
            threads=config.threads
        )
        
        self.report_generator = ReportGenerator(
            quality_output_dir=config.quality_output_dir,
            image_output_dir=config.image_output_dir,
            report_output_dir=config.report_output_dir
        )
    
    def validate_directories(self) -> bool:
        """Validate that all required directories exist and contain valid files."""
        self.logger.info("Validating directories and input files...")
        
        # Check if directories exist
        dirs_to_check = [
            self.config.nlte_models_dir,
            self.config.z_scale_models_dir,
            self.config.quality_output_dir,
            self.config.image_output_dir,
            self.config.report_output_dir
        ]
        
        for directory in dirs_to_check:
            if not os.path.exists(directory):
                try:
                    os.makedirs(directory, exist_ok=True)
                    self.logger.info(f"Created directory: {directory}")
                except Exception as e:
                    self.logger.error(f"Failed to create directory {directory}: {e}")
                    return False
        
        # Check for original spectrum file
        spectrum_file = self.config.original_spectrum_file
        if not os.path.exists(spectrum_file):
            self.logger.error(f"Original spectrum file not found: {spectrum_file}")
            return False
        
        # Check NLTE models directory
        try:
            self.logger.info("Validating NLTE models directory...")
            if not self._validate_nlte_models():
                return False
        except Exception as e:
            self.logger.error(f"Error validating NLTE models: {e}")
            return False
        
        # Check Z-scale models directory if needed
        if not self.config.skip_z_scale_min:
            try:
                self.logger.info("Validating Z-scale models directory...")
                if not self._validate_z_scale_models():
                    return False
            except Exception as e:
                self.logger.error(f"Error validating Z-scale models: {e}")
                return False
        
        self.logger.info("Directory validation completed successfully.")
        return True
    
    def _validate_nlte_models(self) -> bool:
        """Validate NLTE models directory."""
        # Check if directory exists and has H5 files
        if not os.path.isdir(self.config.nlte_models_dir):
            self.logger.error(f"NLTE models directory does not exist: {self.config.nlte_models_dir}")
            return False
        
        h5_files = [f for f in os.listdir(self.config.nlte_models_dir) if f.endswith('.h5') and 'nlte' in f]

        if not h5_files:
            self.logger.error(f"No NLTE model files found in {self.config.nlte_models_dir}")
            return False
        
        # Check for consistent z_scale
        z_scales = set()
        teff_logg_pairs = set()
        
        for file in h5_files:
            try:
                # filename format: nlte{teff}-{logg}+{zscale in \d.\d format}.{model_version}.h5
                teff = file.split('-')[0].replace('nlte', '')
                logg = file.split('-')[1].split('+')[0]
                z_scale = file.split('+')[1].split('.')[0] + '.' + file.split('+')[1].split('.')[1].split('.')[0]

                z_scales.add(z_scale)
                teff_logg_pair = (teff, logg)
                
                if teff_logg_pair in teff_logg_pairs:
                    self.logger.error(f"Duplicate teff-logg pair found: {teff_logg_pair}")
                    return False
                
                teff_logg_pairs.add(teff_logg_pair)
                
            except (IndexError, ValueError) as e:
                self.logger.error(f"Invalid NLTE model filename format: {file} - {e}")
                return False
        
        if len(z_scales) > 1:
            self.logger.error(f"Multiple z-scales found in NLTE models directory: {z_scales}")
            return False
        
        self.logger.info(f"Found {len(h5_files)} valid NLTE model files with z-scale: {list(z_scales)[0]}")
        return True
    
    def _validate_z_scale_models(self) -> bool:
        """Validate Z-scale models directory."""
        if not os.path.isdir(self.config.z_scale_models_dir):
            self.logger.error(f"Z-scale models directory does not exist: {self.config.z_scale_models_dir}")
            return False
        
        h5_files = [f for f in os.listdir(self.config.z_scale_models_dir) if f.endswith('.h5') and 'nlte' in f]
        if not h5_files:
            self.logger.error(f"No Z-scale model files found in {self.config.z_scale_models_dir}")
            return False
        
        # Check if we have enough files for meaningful prediction
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
                
                teff_logg_pairs[key].append(z_scale)
                
            except (IndexError, ValueError) as e:
                self.logger.error(f"Invalid Z-scale model filename format: {file} - {e}")
                return False
        
        # Check if any teff-logg pair has at least 4 different z-scales
        has_enough_points = False
        for key, z_scales in teff_logg_pairs.items():
            if len(z_scales) >= 4:
                has_enough_points = True
                break
        
        if not has_enough_points:
            self.logger.error("Z-scale analysis requires at least one teff-logg pair with 4+ different z-scales")
            return False
        
        self.logger.info(f"Found {len(h5_files)} valid Z-scale model files across {len(teff_logg_pairs)} teff-logg pairs")
        return True
    
    def run_pipeline(self) -> bool:
        """Run the complete pipeline."""
        self.logger.info(f"Starting PHOENIX analysis pipeline - {timestamp}")
        
        # Step 1: Validate directories
        if not self.validate_directories():
            self.logger.error("Directory validation failed. Exiting.")
            return False
        
        # Step 2: Calculate quality values
        self.logger.info("Step 2: Calculating quality values...")
        quality_results = self.quality_calculator.run()
        if not quality_results:
            self.logger.error("Quality calculation failed. Exiting.")
            return False
        
        # Step 3: Find global minimum in heatmap
        self.logger.info("Step 3: Finding global minimum in heatmap...")
        minimum_results = self.heatmap_optimizer.run(quality_results)
        if not minimum_results:
            self.logger.error("Heatmap optimization failed. Exiting.")
            return False
        
        # Step 4: Perform periodicity analysis
        prediction_results = None
        if not self.config.skip_prediction:
            self.logger.info("Step 4a: Performing standard periodicity analysis...")
            periodicity_results = self.periodicity_detector.run(quality_results)
            if not periodicity_results:
                self.logger.warning("Standard periodicity analysis failed, continuing with pipeline.")
            
            # Step 4b: Perform neural periodicity analysis if enabled
            if not self.config.skip_neural_prediction:
                self.logger.info("Step 4b: Performing neural periodicity analysis...")
                prediction_results = self.neural_periodicity_detector.run(quality_results)
                if not prediction_results:
                    self.logger.warning("Neural periodicity analysis failed, continuing with pipeline.")
        
        # Step 5: Analyze z-scale variation if enabled
        z_scale_results = None
        if not self.config.skip_z_scale_min:
            self.logger.info("Step 5: Analyzing z-scale variation...")
            z_scale_results = self.z_scale_optimizer.run()
            if not z_scale_results:
                self.logger.warning("Z-scale optimization failed, continuing with pipeline.")
        
        # Step 6: Generate comprehensive report
        self.logger.info("Step 6: Generating comprehensive report...")
        report_success = self.report_generator.run(
            quality_results=quality_results,
            minimum_results=minimum_results,
            periodicity_results=prediction_results,
            z_scale_results=z_scale_results,
            timestamp=timestamp
        )
        
        if not report_success:
            self.logger.warning("Report generation failed.")
        
        self.logger.info(f"PHOENIX analysis pipeline completed - {timestamp}")
        return True


def parse_args() -> PipelineConfig:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="PHOENIX Stellar Model Analysis Pipeline")
    
    # Check if config file is provided
    # We need to do a preliminary parse to check for config file
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", help="Path to JSON config file containing pipeline parameters")
    pre_args, _ = pre_parser.parse_known_args()
    
    # Load config file if provided
    config_data = {}
    if pre_args.config:
        try:
            with open(pre_args.config, 'r') as f:
                config_data = json.load(f)
            print(f"Loaded configuration from {pre_args.config}")
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Required arguments - only required if config file is not provided
    required_args = not bool(config_data)
    
    # Directory paths - required unless config is provided
    parser.add_argument("--nlte-dir", required=required_args, help="Local directory path for NLTE models")
    parser.add_argument("--z-scale-dir", required=required_args, help="Local directory path for NLTE models with different z-scales")
    parser.add_argument("--quality-dir", required=required_args, help="Local directory path for storing calculated quality value JSONs")
    parser.add_argument("--image-dir", required=required_args, help="Local directory path for storing image outputs")
    parser.add_argument("--report-dir", required=required_args, help="Local directory path for storing reports and predictions")
    
    # Optional configuration file
    parser.add_argument("--config", help="Path to JSON config file containing pipeline parameters")
    
    # Processing options
    parser.add_argument("--recalculate-all", action="store_true", help="Recalculate quality for all NLTE models")
    parser.add_argument("--skip-z-scale-min", action="store_true", help="Skip z-scale minimum calculation")
    parser.add_argument("--skip-prediction", action="store_true", help="Skip prediction analysis")
    parser.add_argument("--skip-neural-prediction", action="store_true", help="Skip neural prediction analysis")
    
    # Additional parameters
    parser.add_argument("--original-spectrum", default="uves_spectra_fomalhaut.csv", 
                      help="Path to original spectrum file (default: uves_spectra_fomalhaut.csv)")
    parser.add_argument("--threads", type=int, default=max(1, os.cpu_count() - 1),
                      help=f"Number of threads to use (default: {max(1, os.cpu_count() - 1)})")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      default="INFO", help="Logging level (default: INFO)")
    
    args = parser.parse_args()
    
    # If config file provided, load it
    config_data = {}
    if args.config:
        try:
            with open(args.config, 'r') as f:
                config_data = json.load(f)
        except Exception as e:
            print(f"Error loading config file: {e}")
            sys.exit(1)
    
    # Create configuration, prioritizing command-line args over config file
    config = PipelineConfig(
        nlte_models_dir=os.path.join(os.getcwd(), args.nlte_dir or config_data.get('nlte_models_dir')),
        z_scale_models_dir=os.path.join(os.getcwd(), args.z_scale_dir or config_data.get('z_scale_models_dir')),
        quality_output_dir=os.path.join(os.getcwd(), args.quality_dir or config_data.get('quality_output_dir')),
        image_output_dir=os.path.join(os.getcwd(), args.image_dir or config_data.get('image_output_dir')),
        report_output_dir=os.path.join(os.getcwd(), args.report_dir or config_data.get('report_output_dir')),
        recalculate_all=args.recalculate_all or config_data.get('recalculate_all', False),
        skip_z_scale_min=args.skip_z_scale_min or config_data.get('skip_z_scale_min', False),
        skip_prediction=args.skip_prediction or config_data.get('skip_prediction', False),
        skip_neural_prediction=args.skip_neural_prediction or config_data.get('skip_neural_prediction', False),
        config_file=args.config,
        original_spectrum_file=os.path.join(os.getcwd(), args.original_spectrum or config_data.get('original_spectrum_file', "uves_spectra_fomalhaut.csv")),
        threads=args.threads or config_data.get('threads', max(1, os.cpu_count() - 1)),
        log_level=args.log_level or config_data.get('log_level', "INFO")
    )
    
    return config


if __name__ == "__main__":
    # Parse arguments
    config = parse_args()

    # Create and run pipeline
    pipeline = PhoenixPipeline(config)
    success = pipeline.run_pipeline()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)