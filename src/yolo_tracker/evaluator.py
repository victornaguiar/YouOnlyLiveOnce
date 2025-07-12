"""
Tracking evaluation using TrackEval metrics.
"""

import os
import sys
import traceback
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import trackeval
except ImportError:
    trackeval = None
    print("Warning: TrackEval not installed. Install with: pip install git+https://github.com/JonathonLuiten/TrackEval.git")


class TrackEvaluator:
    """Evaluates tracking results using TrackEval metrics."""
    
    def __init__(self, gt_folder: str, trackers_folder: str, 
                 use_parallel: bool = False, num_parallel_cores: int = 1):
        """
        Initialize the evaluator.
        
        Args:
            gt_folder: Path to ground truth folder
            trackers_folder: Path to tracker results folder
            use_parallel: Whether to use parallel processing
            num_parallel_cores: Number of parallel cores to use
        """
        self.gt_folder = Path(gt_folder)
        self.trackers_folder = Path(trackers_folder)
        self.use_parallel = use_parallel
        self.num_parallel_cores = num_parallel_cores
        
        if trackeval is None:
            raise ImportError("TrackEval is required for evaluation. Install with: "
                            "pip install git+https://github.com/JonathonLuiten/TrackEval.git")
    
    def evaluate_tracker(self, tracker_name: str, benchmark: str = "MOT17", 
                        split: str = "train", metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a tracker using TrackEval.
        
        Args:
            tracker_name: Name of the tracker to evaluate
            benchmark: Benchmark name (e.g., "MOT17", "MOT20")
            split: Dataset split ("train", "test", "val")
            metrics: List of metrics to compute (default: ["HOTA", "CLEAR", "Identity"])
            
        Returns:
            Dict containing evaluation results
        """
        if metrics is None:
            metrics = ["HOTA", "CLEAR", "Identity"]
            
        try:
            # Configure TrackEval
            eval_config = {
                'USE_PARALLEL': self.use_parallel,
                'NUM_PARALLEL_CORES': self.num_parallel_cores,
                'BREAK_ON_ERROR': True,
                'RETURN_ON_ERROR': False,
                'LOG_ON_ERROR': str(Path.cwd() / 'error_log.txt'),
                'PRINT_RESULTS': True,
                'PRINT_ONLY_COMBINED': False,
                'PRINT_CONFIG': True,
                'TIME_PROGRESS': True,
                'DISPLAY_LESS_PROGRESS': True,
                'OUTPUT_SUMMARY': True,
                'OUTPUT_EMPTY_CLASSES': True,
                'OUTPUT_DETAILED': True,
                'PLOT_CURVES': False,
            }
            
            # Configure dataset
            dataset_config = {
                'GT_FOLDER': str(self.gt_folder),
                'TRACKERS_FOLDER': str(self.trackers_folder),
                'OUTPUT_FOLDER': None,  # Use default
                'TRACKERS_TO_EVAL': [tracker_name],
                'CLASSES_TO_EVAL': ['pedestrian'],  # Default for MOT
                'BENCHMARK': benchmark,
                'SPLIT_TO_EVAL': split,
                'INPUT_AS_ZIP': False,
                'PRINT_CONFIG': True,
                'DO_PREPROC': True,
                'TRACKER_SUB_FOLDER': '',
                'OUTPUT_SUB_FOLDER': '',
                'TRACKER_DISPLAY_NAMES': None,
                'SEQMAP_FOLDER': None,
                'SEQMAP_FILE': None,
                'SEQ_INFO': None,
                'GT_LOC_FORMAT': '{gt_folder}/{seq}/gt/gt.txt',
                'SKIP_SPLIT_FOL': True,
            }
            
            # Configure metrics
            metrics_config = {
                'METRICS': metrics,
                'THRESHOLD': 0.5,
                'PRINT_CONFIG': True,
            }
            
            # Create evaluator
            evaluator = trackeval.Evaluator(eval_config)
            
            # Get dataset
            if benchmark == "MOT17":
                dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
            else:
                # Generic dataset
                dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
                
            # Get metrics
            metrics_list = []
            for metric in metrics:
                if metric == "HOTA":
                    metrics_list.append(trackeval.metrics.HOTA(metrics_config))
                elif metric == "CLEAR":
                    metrics_list.append(trackeval.metrics.CLEAR(metrics_config))
                elif metric == "Identity":
                    metrics_list.append(trackeval.metrics.Identity(metrics_config))
                    
            # Run evaluation
            output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
            
            return {
                'success': True,
                'results': output_res,
                'messages': output_msg,
                'tracker_name': tracker_name,
                'metrics': metrics
            }
            
        except Exception as e:
            print(f"Error during evaluation: {e}")
            traceback.print_exc()
            return {
                'success': False,
                'error': str(e),
                'tracker_name': tracker_name,
                'metrics': metrics
            }
    
    def evaluate_multiple_trackers(self, tracker_names: List[str], **kwargs) -> Dict[str, Any]:
        """
        Evaluate multiple trackers.
        
        Args:
            tracker_names: List of tracker names to evaluate
            **kwargs: Additional arguments passed to evaluate_tracker
            
        Returns:
            Dict containing results for all trackers
        """
        results = {}
        
        for tracker_name in tracker_names:
            print(f"\n--- Evaluating tracker: {tracker_name} ---")
            results[tracker_name] = self.evaluate_tracker(tracker_name, **kwargs)
            
        return results
    
    def print_summary(self, results: Dict[str, Any]):
        """
        Print a summary of evaluation results.
        
        Args:
            results: Results from evaluate_tracker or evaluate_multiple_trackers
        """
        if isinstance(results, dict) and 'tracker_name' in results:
            # Single tracker results
            self._print_single_tracker_summary(results)
        else:
            # Multiple tracker results
            for tracker_name, tracker_results in results.items():
                print(f"\n=== {tracker_name} ===")
                self._print_single_tracker_summary(tracker_results)
                
    def _print_single_tracker_summary(self, results: Dict[str, Any]):
        """Print summary for a single tracker."""
        if not results['success']:
            print(f"Evaluation failed: {results.get('error', 'Unknown error')}")
            return
            
        print(f"Tracker: {results['tracker_name']}")
        print(f"Metrics evaluated: {', '.join(results['metrics'])}")
        
        # Extract key metrics if available
        if 'results' in results and results['results']:
            try:
                # This would need to be adapted based on TrackEval output structure
                res = results['results']
                print("Evaluation completed successfully.")
                print("Detailed results available in the results dictionary.")
            except Exception as e:
                print(f"Error extracting metrics: {e}")
                
    def validate_setup(self) -> bool:
        """
        Validate that the evaluation setup is correct.
        
        Returns:
            bool: True if setup is valid, False otherwise
        """
        issues = []
        
        # Check ground truth folder
        if not self.gt_folder.exists():
            issues.append(f"Ground truth folder not found: {self.gt_folder}")
            
        # Check trackers folder
        if not self.trackers_folder.exists():
            issues.append(f"Trackers folder not found: {self.trackers_folder}")
            
        # Check TrackEval installation
        if trackeval is None:
            issues.append("TrackEval not installed")
            
        if issues:
            print("Validation issues found:")
            for issue in issues:
                print(f"  - {issue}")
            return False
            
        print("Evaluation setup validation passed.")
        return True
        
    def setup_evaluation_folders(self, sequences: List[str], create_gt_structure: bool = True):
        """
        Set up the folder structure required for TrackEval.
        
        Args:
            sequences: List of sequence names
            create_gt_structure: Whether to create the GT folder structure
        """
        # Create main folders
        self.gt_folder.mkdir(parents=True, exist_ok=True)
        self.trackers_folder.mkdir(parents=True, exist_ok=True)
        
        if create_gt_structure:
            for seq in sequences:
                seq_gt_folder = self.gt_folder / seq / "gt"
                seq_gt_folder.mkdir(parents=True, exist_ok=True)
                
                # Create empty gt.txt if it doesn't exist
                gt_file = seq_gt_folder / "gt.txt"
                if not gt_file.exists():
                    gt_file.touch()
                    
        print(f"Evaluation folders set up:")
        print(f"  GT folder: {self.gt_folder}")
        print(f"  Trackers folder: {self.trackers_folder}")