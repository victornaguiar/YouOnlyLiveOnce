"""
Dataset management utilities for tracking evaluation.
"""

import os
import shutil
import configparser
import cv2
from pathlib import Path
from typing import List, Optional, Dict


class DatasetManager:
    """Manages dataset files and sequence information for tracking evaluation."""
    
    def __init__(self, dataset_root: str):
        """
        Initialize dataset manager.
        
        Args:
            dataset_root: Root directory of the dataset
        """
        self.dataset_root = Path(dataset_root)
        self.sequences_dir = self.dataset_root / "sequences"
        self.detections_dir = self.dataset_root / "detections"
        self.results_dir = self.dataset_root / "tracking_results"
        
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.sequences_dir, self.detections_dir, self.results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
    def generate_sequence_info_files(self) -> bool:
        """
        Generate seqinfo.ini files for sequences that don't have them.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.sequences_dir.exists():
                print(f"ERROR: Sequences directory not found: {self.sequences_dir}")
                return False
                
            sequence_folders = sorted([d for d in self.sequences_dir.iterdir() 
                                     if d.is_dir()])
            
            if not sequence_folders:
                print(f"ERROR: No sequence folders found in {self.sequences_dir}")
                return False
                
            print(f"Found {len(sequence_folders)} sequence folders to process.")
            
            for seq_path in sequence_folders:
                ini_file_path = seq_path / 'seqinfo.ini'
                
                if ini_file_path.exists():
                    print(f"  - Skipping {seq_path.name}: seqinfo.ini already exists.")
                    continue
                    
                # Find image files in the sequence directory
                img_files = self._find_image_files(seq_path)
                
                if not img_files:
                    print(f"  - Skipping {seq_path.name}: No image files found.")
                    continue
                    
                # Get image dimensions from first image
                first_img_path = seq_path / img_files[0]
                try:
                    img = cv2.imread(str(first_img_path))
                    if img is None:
                        print(f"  - Skipping {seq_path.name}: Could not read {first_img_path}")
                        continue
                        
                    img_height, img_width = img.shape[:2]
                except Exception as e:
                    print(f"  - Error reading {first_img_path}: {e}")
                    continue
                    
                # Create seqinfo.ini
                self._create_seqinfo_file(ini_file_path, seq_path.name, 
                                        len(img_files), img_width, img_height)
                
                print(f"  - Created seqinfo.ini for {seq_path.name}")
                
            return True
            
        except Exception as e:
            print(f"Error generating sequence info files: {e}")
            return False
            
    def _find_image_files(self, directory: Path) -> List[str]:
        """Find image files in a directory."""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        img_files = []
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                img_files.append(file_path.name)
                
        return sorted(img_files)
    
    def _create_seqinfo_file(self, ini_path: Path, seq_name: str, 
                           seq_length: int, img_width: int, img_height: int):
        """Create a seqinfo.ini file."""
        config = configparser.ConfigParser()
        config['Sequence'] = {
            'name': seq_name,
            'imDir': 'img1',  # Standard MOT format
            'frameRate': '30',  # Default frame rate
            'seqLength': str(seq_length),
            'imWidth': str(img_width),
            'imHeight': str(img_height),
            'imExt': '.jpg'  # Default extension
        }
        
        with open(ini_path, 'w') as f:
            config.write(f)
            
    def copy_data_locally(self, source_sequences: str, source_detections: str, 
                         source_results: str, local_root: str) -> bool:
        """
        Copy dataset files to local directory for faster processing.
        
        Args:
            source_sequences: Source sequences directory
            source_detections: Source detections directory
            source_results: Source results directory
            local_root: Local root directory to copy to
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            local_root = Path(local_root)
            
            # Define local paths
            local_sequences = local_root / "sequences"
            local_detections = local_root / "gt_files"
            local_results = local_root / "tracking_results"
            
            # Copy sequences
            if os.path.exists(source_sequences):
                print(f"Copying sequences from {source_sequences}...")
                if local_sequences.exists():
                    shutil.rmtree(local_sequences)
                shutil.copytree(source_sequences, local_sequences)
                print("...Sequences copied successfully.")
            
            # Copy detections (ground truth)
            if os.path.exists(source_detections):
                print(f"Copying detections from {source_detections}...")
                if local_detections.exists():
                    shutil.rmtree(local_detections)
                shutil.copytree(source_detections, local_detections)
                print("...Detections copied successfully.")
            
            # Copy results
            if os.path.exists(source_results):
                print(f"Copying results from {source_results}...")
                if local_results.exists():
                    shutil.rmtree(local_results)
                shutil.copytree(source_results, local_results)
                print("...Results copied successfully.")
                
            return True
            
        except Exception as e:
            print(f"Error copying data locally: {e}")
            return False
            
    def validate_dataset_structure(self) -> Dict[str, bool]:
        """
        Validate that the dataset has the expected structure.
        
        Returns:
            Dict[str, bool]: Validation results for different components
        """
        validation = {
            'sequences_dir_exists': self.sequences_dir.exists(),
            'detections_dir_exists': self.detections_dir.exists(),
            'results_dir_exists': self.results_dir.exists(),
            'has_sequences': False,
            'has_detections': False,
            'has_results': False
        }
        
        # Check for sequences
        if validation['sequences_dir_exists']:
            sequences = list(self.sequences_dir.iterdir())
            validation['has_sequences'] = len(sequences) > 0
            
        # Check for detections
        if validation['detections_dir_exists']:
            detections = list(self.detections_dir.glob('*.txt'))
            validation['has_detections'] = len(detections) > 0
            
        # Check for results
        if validation['results_dir_exists']:
            results = list(self.results_dir.glob('*.txt'))
            validation['has_results'] = len(results) > 0
            
        return validation
        
    def list_sequences(self) -> List[str]:
        """List all available sequences."""
        if not self.sequences_dir.exists():
            return []
            
        return [d.name for d in self.sequences_dir.iterdir() if d.is_dir()]
        
    def get_sequence_info(self, sequence_name: str) -> Optional[Dict]:
        """
        Get information about a specific sequence.
        
        Args:
            sequence_name: Name of the sequence
            
        Returns:
            Dict with sequence information or None if not found
        """
        seq_path = self.sequences_dir / sequence_name
        ini_path = seq_path / 'seqinfo.ini'
        
        if not ini_path.exists():
            return None
            
        config = configparser.ConfigParser()
        config.read(ini_path)
        
        if 'Sequence' not in config:
            return None
            
        return dict(config['Sequence'])