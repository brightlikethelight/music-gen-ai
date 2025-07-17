#!/usr/bin/env python3
"""
Setup script for MusicGen Unified
Harvard CS 109B Final Project
"""

from setuptools import setup, find_packages

# Read the README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="musicgen-unified",
    version="2.0.1",
    author="Bright Liu",
    author_email="brightliu@college.harvard.edu",
    description="Unified MusicGen interface for AI music generation - Harvard CS 109B Final Project",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/brightlikethelight/music-gen-ai",
    project_urls={
        "Bug Reports": "https://github.com/brightlikethelight/music-gen-ai/issues",
        "Documentation": "https://github.com/brightlikethelight/music-gen-ai/tree/main/docs",
        "CS 109B Project": "https://github.com/brightlikethelight/music-gen-ai/tree/main/docs/cs109b",
        "Source": "https://github.com/brightlikethelight/music-gen-ai",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Sound/Audio :: Sound Synthesis",
        "Topic :: Education",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.26.0,<2.0",
        "scipy>=1.14.0,<2.0",
        "torch>=2.2.0",
        "transformers>=4.43.0",
        "scikit-learn>=1.3.0",
        "soundfile>=0.12.0",
        "librosa>=0.10.0",
        "pydub>=0.25.0",
        "typer[all]>=0.9.0",
        "rich>=13.0.0",
        "pandas>=2.0.0",
    ],
    entry_points={
        "console_scripts": [
            "musicgen=musicgen.cli:app",
        ],
    },
    include_package_data=True,
    package_data={
        "musicgen": ["py.typed"],
    },
    keywords="music generation ai machine-learning deep-learning pytorch transformers audio musicgen harvard cs109b",
)