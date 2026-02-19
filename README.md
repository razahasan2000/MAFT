# MAFT: Multimodal Attention Fusion Transformer for Maritime Anomaly Detection

This repository contains the official implementation of **MAFT** (Maritime Anomaly Fusion Transformer), a robust multimodal framework for detecting anomalies in maritime surveillance data. The model fuses **Aerial Imagery**, **SAR (Synthetic Aperture Radar)**, **Acoustic Spectrograms**, and **AIS (Automatic Identification System)** data using a self-attention mechanism with modality dropout.

## 🚀 Key Features
*   **Multimodal Fusion**: Integrates 4 heterogeneous data sources (Visual, Radar, Audio, Telemetry).
*   **Interpretation**: Self-attention weights provide interpretability for anomaly scores.
*   **Robustness**: Trained with **Modality Dropout** to handle missing sensor data during inference.
*   **Efficiency**: Optimized for edge deployment (~26ms latency on GPU).

## 📂 Repository Structure
```
MAFT-v3/
├── data/
│   └── synthetic_generation/ # Scripts to generate the synthetic maritime dataset
├── models/
│   ├── maft_v3.py           # Core MAFT architecture
│   └── baselines/           # Baseline models (CNN, MLP, Ensemble)
├── training/
│   ├── train.py             # Main training script
│   ├── dataset.py           # PyTorch Dataset implementation
│   ├── utils.py             # Helper functions (Metrics, EMA, MixUp)
│   └── config.yaml          # Hyperparameter configuration
└── analysis/
    └── compare_baselines.py # Script to partial-replicate the paper's comparison table
```

## 🛠️ Installation
1.  Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MAFT-v3.git
    cd MAFT-v3
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## 📊 Data Generation
The dataset is synthetic, designed to simulate realistic maritime scenarios.
To generate the dataset (approx. 2.5GB):
```bash
python data/synthetic_generation/generate_samples.py
```
This will create `synthetic_data_v2/` in the root directory.

## 🏋️ Training
To train the MAFT v3 model from scratch:
```bash
python training/train.py
```
*   **Config**: Modify `training/config.yaml` to adjust batch size, learning rate, or epochs.
*   **Output**: Best weights will be saved to `working/maft_advanced_v3.pth`.

## 📉 Evaluation
To run the comprehensive baseline comparison (Accuracy, F1, ECE, Latency):
```bash
python analysis/compare_baselines.py
```
Results will be saved to `working/results/`.

## 📜 Citation
If you use this code in your research, please cite our paper:
```bibtex
@article{hasan2026maft,
  title={Late-Fusion of Heterogeneous Maritime Data using Self-Attention for Interpretable Anomaly Detection},
  author={Hasan, Raza and Ahmad, Shakeel and Gocer, Ismet and Bhuiyan, Zakirul},
  journal={Computers, Materials & Continua},
  year={2026}
}
```

## 📄 License
MIT License
