# CMPE 492: Explainable Multimodal Triage for Highly Imbalanced Clinical Datasets

This repository contains the code for my CMPE 492 graduation project. It implements a multimodal machine learning framework for skin cancer triage using the ISIC 2024 dataset, featuring hard negative mining and FAISS-based similar case retrieval.

## Installation
This project uses `uv` for dependency management.
Run `uv sync` to install all required dependencies.

## Dataset Preparation
1. Download the ISIC 2024 dataset from the [ISIC Archive](https://www.isic-archive.com/).
2. Place the downloaded images and metadata under the `data/ISIC` folder.

The expected directory structure is:
```text
data/
└── ISIC/
    ├── ISIC_2024_Training_Input/
    ├── ISIC_2024_Training_GroundTruth.csv
    └── metadata.csv
```

## Notebooks
- The notebooks related to melanoma studies with the data from ISIC are under `notebooks/melanoma` folder. 
- Additionally, the `notebooks/pcnl` folder contains the previous works on Pediatric PCNL prediction.