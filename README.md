# DeepSort from Ground Truth

This repository contains a Python script for performing object tracking using the DeepSort algorithm on the SoccerNet dataset, utilizing ground truth bounding boxes as input.

## Overview

The `deepsort_from_gt.py` script allows you to:
1. **Initialize DeepSort**: Sets up the tracker using the `deep_sort_realtime` library
2. **Load Ground Truth Data**: Parses MOT-formatted `.txt` files from your dataset
3. **Process Image Sequences**: Iterates through image frames of a specified sequence (e.g., `SNMOT-116`)
4. **Run Tracking**: Feeds ground truth detections into DeepSort to get tracking IDs
5. **Generate Output Video**: Creates and saves a `.mp4` video with tracked bounding boxes and IDs

## Installation

### For Google Colab

1. **Install PyTorch and dependencies:**
   ```bash
   # Install compatible PyTorch and Torchvision for Colab
   !pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html
   
   # Install other required packages
   !pip install deep-sort-realtime opencv-python pandas tqdm
   ```

2. **Clone or download this repository:**
   ```bash
   # If cloning from GitHub
   !git clone https://github.com/victornaguiar/YouOnlyLiveOnce.git
   %cd YouOnlyLiveOnce
   ```

3. **Mount Google Drive (if your data is stored there):**
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

### For Local Installation

```bash
pip install torch torchvision deep-sort-realtime opencv-python pandas tqdm
```

## Usage

### Basic Usage

Run the script with default parameters (processes SNMOT-116 sequence):

```bash
python deepsort_from_gt.py
```

### Custom Parameters

```bash
python deepsort_from_gt.py \
    --sequence SNMOT-060 \
    --data_dir /path/to/your/data \
    --output /path/to/output_video.mp4 \
    --max_age 50 \
    --n_init 3 \
    --embedder mobilenet
```

### Parameters

- `--sequence`: Sequence name (default: SNMOT-116)
- `--data_dir`: Base directory containing the dataset (default: /content/drive/MyDrive/SOCCER_DATA)
- `--output`: Output video path (default: /content/{sequence}_tracked_output.mp4)
- `--max_age`: Maximum age for tracks (default: 30)
- `--n_init`: Number of consecutive detections before track is confirmed (default: 3)
- `--max_iou_distance`: Maximum IOU distance for track matching (default: 0.7)
- `--embedder`: Feature embedder to use (default: mobilenet)

## Dataset Structure

The script expects the following directory structure:

```
data_dir/
├── sequences/
│   └── SNMOT-116/
│       ├── 000001.jpg
│       ├── 000002.jpg
│       └── ...
└── detections/
    └── SNMOT-116.txt
```

The ground truth file should be in MOT format:
```
frame_id,track_id,x,y,width,height,confidence,class,visibility
1,1,100,100,50,50,0.9,1,1
2,1,105,105,50,50,0.9,1,1
...
```

## Features

- **Flexible Input**: Works with various image formats (JPG, PNG, BMP, TIFF)
- **MOT Format Support**: Reads standard MOT challenge format ground truth files
- **Configurable Tracking**: Adjustable DeepSort parameters
- **Video Output**: Generates MP4 videos with tracking overlays
- **Progress Tracking**: Shows progress bar during processing
- **Error Handling**: Graceful handling of missing files and other errors

## Example Output

The script generates a video file showing:
- Original frames with bounding boxes
- Unique track IDs for each detected object
- Frame numbers and track count overlays
- Color-coded tracks for easy visualization

## Troubleshooting

### Common Issues

1. **"No module named 'torch'" Error:**
   - Install PyTorch: `pip install torch torchvision`
   - In Colab: Use the CUDA-compatible version shown in installation instructions

2. **"Ground truth file not found" Error:**
   - Check the `--data_dir` parameter
   - Ensure the MOT format file exists in the `detections/` subdirectory

3. **"No image files found" Error:**
   - Verify the sequence name matches the folder name
   - Check that images are in supported formats (JPG, PNG, etc.)

### Performance Tips

- Use `--embedder mobilenet` for faster processing
- Reduce `--max_age` if you have short sequences
- Increase `--n_init` for more stable tracking in crowded scenes

## Citation

If you use this script in your research, please cite:

```bibtex
@misc{deepsort_from_gt,
  title={DeepSort Tracking from Ground Truth Detections},
  author={Your Name},
  year={2024},
  url={https://github.com/victornaguiar/YouOnlyLiveOnce}
}
```