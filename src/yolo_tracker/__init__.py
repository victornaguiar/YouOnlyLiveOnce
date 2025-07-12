"""
YOLO Tracker - Multi-object tracking using YOLO and BotSort.

A Python package for object detection and tracking in videos using YOLOv8 and BotSort tracker.
Supports evaluation using TrackEval metrics.
"""

__version__ = "1.0.0"
__author__ = "Victor Naguiar"

# Import modules with graceful error handling for missing dependencies
_available_modules = []

try:
    from .tracker import YOLOTracker
    _available_modules.append("YOLOTracker")
except ImportError as e:
    YOLOTracker = None
    print(f"Warning: YOLOTracker not available due to missing dependencies: {e}")

try:
    from .evaluator import TrackEvaluator
    _available_modules.append("TrackEvaluator")
except ImportError as e:
    TrackEvaluator = None
    print(f"Warning: TrackEvaluator not available due to missing dependencies: {e}")

try:
    from .dataset import DatasetManager
    _available_modules.append("DatasetManager")
except ImportError as e:
    DatasetManager = None
    print(f"Warning: DatasetManager not available due to missing dependencies: {e}")

__all__ = _available_modules