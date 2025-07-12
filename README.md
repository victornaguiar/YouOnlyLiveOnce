# YOLO Tracker

A Python package for multi-object tracking using YOLO detection and BotSort tracker. This project converts the original Jupyter notebook implementation into a proper Python package with modular design, CLI interface, and comprehensive evaluation capabilities.

## Features

- **Object Detection**: YOLO-based object detection (YOLOv8)
- **Multi-Object Tracking**: BotSort tracker for robust object tracking
- **Custom Dataset Support**: Process custom datasets with pre-computed detections
- **Evaluation**: Integration with TrackEval for comprehensive tracking metrics
- **Configuration Management**: Flexible YAML/JSON configuration system
- **CLI Interface**: Command-line tools for all operations
- **Modular Design**: Clean separation of concerns with well-defined modules

## Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/victornaguiar/YouOnlyLiveOnce.git
cd YouOnlyLiveOnce

# Install in development mode
pip install -e .

# Or install for evaluation features
pip install -e ".[eval]"
```

### Dependencies

The package requires Python 3.8+ and the following main dependencies:

- PyTorch >= 2.0.0
- Ultralytics (YOLOv8)
- BoxMOT (BotSort tracker)
- OpenCV
- NumPy
- TrackEval (optional, for evaluation)

See `requirements.txt` for the complete list.

## Quick Start

### 1. Basic Video Tracking

```python
from yolo_tracker import YOLOTracker

# Initialize tracker
tracker = YOLOTracker(model_path='yolov8n.pt')

# Track objects in video
success = tracker.track_video('input_video.mp4', 'tracked_output.mp4')
```

### 2. Using the CLI

```bash
# Track a video
yolo-tracker track input_video.mp4 tracked_output.mp4

# Create a test video
yolo-tracker create-test-video test_video.mp4 --duration 10

# Run a complete demo
yolo-tracker demo --output demo_results/
```

### 3. Custom Dataset Processing

```python
from yolo_tracker import YOLOTracker, DatasetManager

# Setup dataset
dataset = DatasetManager('/path/to/dataset')
dataset.setup_directories()
dataset.generate_sequence_info_files()

# Track with custom detections
tracker = YOLOTracker()
tracker.track_with_custom_detections(
    'detections.txt',
    '/path/to/sequence',
    'tracking_results.txt'
)
```

### 4. Evaluation

```python
from yolo_tracker import TrackEvaluator

# Initialize evaluator
evaluator = TrackEvaluator(
    gt_folder='/path/to/ground_truth',
    trackers_folder='/path/to/tracker_results'
)

# Evaluate tracker
results = evaluator.evaluate_tracker(
    'my_tracker',
    metrics=['HOTA', 'CLEAR', 'Identity']
)

evaluator.print_summary(results)
```

## Command Line Interface

The package provides a comprehensive CLI with the following commands:

### Tracking Commands

```bash
# Track objects in video
yolo-tracker track input.mp4 output.mp4 --model yolov8s.pt

# Track using pre-computed detections
yolo-tracker track-detections detections.txt sequence_dir/ results.txt
```

### Evaluation Commands

```bash
# Evaluate tracking results
yolo-tracker evaluate gt_folder/ tracker_folder/ my_tracker \
    --metrics HOTA CLEAR Identity --benchmark MOT17
```

### Dataset Management

```bash
# Generate sequence info files
yolo-tracker dataset generate-seqinfo /path/to/dataset

# Validate dataset structure
yolo-tracker dataset validate /path/to/dataset

# Copy data locally (for faster processing)
yolo-tracker dataset copy-local source_seq/ source_det/ source_res/ local_dir/
```

### Configuration Management

```bash
# Show current configuration
yolo-tracker config show

# Create default configuration file
yolo-tracker config create my_config.yaml
```

### Demo and Testing

```bash
# Run complete demo
yolo-tracker demo --output demo_results/

# Create test video
yolo-tracker create-test-video test.mp4 --width 1280 --height 720 --duration 10
```

## Configuration

The package uses a flexible configuration system supporting both YAML and JSON formats:

```yaml
# config.yaml
model:
  yolo_model: "yolov8n.pt"
  tracker_type: "botsort"
  device: "auto"
  confidence_threshold: 0.5

tracker:
  botsort:
    track_high_thresh: 0.5
    track_low_thresh: 0.1
    new_track_thresh: 0.6
    track_buffer: 30

dataset:
  sequences_dir: "sequences"
  detections_dir: "detections"
  results_dir: "tracking_results"

evaluation:
  metrics: ["HOTA", "CLEAR", "Identity"]
  benchmark: "MOT17"
  split: "train"
```

Load configuration:

```python
from yolo_tracker.config import Config

config = Config('config.yaml')
# or use environment variables
config = Config()  # Will use YOLO_* environment variables
```

## Project Structure

```
YouOnlyLiveOnce/
├── src/yolo_tracker/           # Main package
│   ├── __init__.py            # Package initialization
│   ├── tracker.py             # YOLO + BotSort tracking
│   ├── evaluator.py           # TrackEval integration
│   ├── dataset.py             # Dataset management
│   ├── config.py              # Configuration management
│   └── cli.py                 # Command-line interface
├── examples/                   # Example scripts
│   ├── basic_tracking.py      # Basic tracking example
│   ├── dataset_processing.py  # Dataset processing example
│   └── configuration_example.py # Configuration example
├── config/                     # Configuration files
├── tests/                      # Unit tests
├── docs/                       # Documentation
├── requirements.txt            # Dependencies
├── setup.py                    # Package setup
└── README.md                   # This file
```

## Examples

The `examples/` directory contains several example scripts:

1. **basic_tracking.py**: Simple video tracking
2. **dataset_processing.py**: Custom dataset processing
3. **configuration_example.py**: Configuration management demo

Run examples:

```bash
cd examples/
python basic_tracking.py
python dataset_processing.py
python configuration_example.py
```

## Original Notebook Features

This package implements all functionality from the original Jupyter notebook:

- ✅ YOLO-based object detection
- ✅ BotSort tracking integration
- ✅ Custom dataset processing
- ✅ TrackEval metrics computation
- ✅ Sequence info generation
- ✅ Ground truth evaluation
- ✅ Test video creation
- ✅ Configurable parameters

### Key Improvements

- **Modular Design**: Clean separation into logical modules
- **CLI Interface**: No need for Jupyter notebook environment
- **Configuration Management**: Flexible config system
- **Error Handling**: Robust error handling and validation
- **Documentation**: Comprehensive documentation and examples
- **Packaging**: Proper Python package with setup.py
- **Portability**: Runs anywhere Python is available (no Colab dependency)

## Dataset Format

The package expects datasets in MOT (Multiple Object Tracking) format:

```
dataset_root/
├── sequences/
│   ├── sequence1/
│   │   ├── img1/           # Images
│   │   └── seqinfo.ini     # Sequence metadata
│   └── sequence2/
├── detections/
│   ├── sequence1.txt       # Detection files
│   └── sequence2.txt
└── tracking_results/
    ├── sequence1.txt       # Tracking results
    └── sequence2.txt
```

Detection file format:
```
<frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for YOLOv8
- [BoxMOT](https://github.com/mikel-brostrom/yolov8_tracking) for BotSort tracker
- [TrackEval](https://github.com/JonathonLuiten/TrackEval) for evaluation metrics
- Original Jupyter notebook implementation by Victor Naguiar

## Support

For questions, issues, or contributions, please:

1. Check the [Issues](https://github.com/victornaguiar/YouOnlyLiveOnce/issues) page
2. Create a new issue if your problem isn't already reported
3. Provide detailed information about your environment and the issue

## Changelog

### v1.0.0 (Initial Release)
- Converted Jupyter notebook to Python package
- Added CLI interface
- Implemented modular design
- Added configuration management
- Added comprehensive examples
- Added documentation