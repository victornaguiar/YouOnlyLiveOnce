#!/usr/bin/env python3
"""
Simple test script to verify package functionality without heavy dependencies.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path to import yolo_tracker
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))


def test_config():
    """Test configuration functionality."""
    print("Testing configuration...")
    
    from yolo_tracker.config import Config
    
    # Test default config
    config = Config()
    assert config.get('model.yolo_model') == 'yolov8n.pt'
    assert config.get('model.device') == 'cpu'
    print("  ✓ Default configuration loaded")
    
    # Test config modification
    config.set('model.confidence_threshold', 0.8)
    assert config.get('model.confidence_threshold') == 0.8
    print("  ✓ Configuration modification works")
    
    # Test config save/load
    with tempfile.NamedTemporaryFile(suffix='.yaml', delete=False) as f:
        temp_config_path = f.name
        
    try:
        config.save_config(temp_config_path)
        
        new_config = Config(temp_config_path)
        assert new_config.get('model.confidence_threshold') == 0.8
        print("  ✓ Configuration save/load works")
        
    finally:
        Path(temp_config_path).unlink(missing_ok=True)
    
    print("Configuration tests passed!")


def test_dataset_manager():
    """Test dataset manager functionality."""
    print("Testing dataset manager...")
    
    from yolo_tracker.dataset import DatasetManager
    
    # Create temporary dataset directory
    with tempfile.TemporaryDirectory() as temp_dir:
        dataset_manager = DatasetManager(temp_dir)
        
        # Test directory setup
        dataset_manager.setup_directories()
        assert dataset_manager.sequences_dir.exists()
        assert dataset_manager.detections_dir.exists()
        assert dataset_manager.results_dir.exists()
        print("  ✓ Directory setup works")
        
        # Test validation
        validation = dataset_manager.validate_dataset_structure()
        assert validation['sequences_dir_exists'] is True
        assert validation['detections_dir_exists'] is True
        assert validation['results_dir_exists'] is True
        print("  ✓ Dataset validation works")
        
        # Test sequence listing (should be empty)
        sequences = dataset_manager.list_sequences()
        assert len(sequences) == 0
        print("  ✓ Sequence listing works")
    
    print("Dataset manager tests passed!")


def test_package_import():
    """Test package import functionality."""
    print("Testing package import...")
    
    import yolo_tracker
    
    # Check package attributes
    assert hasattr(yolo_tracker, '__version__')
    assert hasattr(yolo_tracker, '__author__')
    assert hasattr(yolo_tracker, '_available_modules')
    print("  ✓ Package attributes present")
    
    # Check available modules (should have DatasetManager and possibly TrackEvaluator)
    available = yolo_tracker._available_modules
    assert 'DatasetManager' in available
    print(f"  ✓ Available modules: {available}")
    
    print("Package import tests passed!")


def main():
    """Run all tests."""
    print("=== Running YOLO Tracker Tests ===\n")
    
    try:
        test_config()
        print()
        
        test_dataset_manager()
        print()
        
        test_package_import()
        print()
        
        print("🎉 All tests passed! The package is working correctly.")
        print("\nTo enable full functionality, install the remaining dependencies:")
        print("  pip install torch torchvision ultralytics boxmot")
        print("  pip install git+https://github.com/JonathonLuiten/TrackEval.git")
        
        return 0
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())