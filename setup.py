"""
Setup script for YOLO Tracker package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README file
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    with open(requirements_path, 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
else:
    requirements = [
        'torch>=2.0.0',
        'torchvision>=0.15.0',
        'numpy>=1.21.0',
        'opencv-python>=4.5.0',
        'scipy>=1.7.0',
        'ultralytics>=8.0.0',
        'boxmot>=10.0.0',
        'pathlib',
        'configparser',
        'collections'
    ]

setup(
    name="yolo-tracker",
    version="1.0.0",
    author="Victor Naguiar",
    author_email="",
    description="Multi-object tracking using YOLO detection and BotSort tracker",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/victornaguiar/YouOnlyLiveOnce",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.900",
        ],
        "eval": [
            "git+https://github.com/JonathonLuiten/TrackEval.git",
        ],
    },
    entry_points={
        "console_scripts": [
            "yolo-tracker=yolo_tracker.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "yolo_tracker": ["*.yaml", "*.json"],
    },
    project_urls={
        "Bug Reports": "https://github.com/victornaguiar/YouOnlyLiveOnce/issues",
        "Source": "https://github.com/victornaguiar/YouOnlyLiveOnce",
    },
)