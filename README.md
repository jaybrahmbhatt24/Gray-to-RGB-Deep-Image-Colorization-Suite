# 🌈 Gray to RGB — Deep Image Colorization Suite  
### *ECCV16 + SIGGRAPH17 Models · PyTorch · Streamlit App (Offline)*  

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)]()
[![Streamlit](https://img.shields.io/badge/Streamlit-App-green.svg)]()
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A complete offline tool for colorizing grayscale images using the **official ECCV16** and **SIGGRAPH17** models by **Richard Zhang et al.**  
Run both models side-by-side, compare outputs visually, download the results, and explore dominant color palettes — all inside an interactive Streamlit UI.

---

# 📌 Table of Contents

- [✨ Overview](#-overview)  
- [🎯 Goal & Problem Definition](#-goal--problem-definition)  
- [📂 Repository Structure](#-repository-structure)  
- [⭐ Features](#-features)  
- [🧰 Tech Stack](#-tech-stack)  
- [📦 Requirements](#-requirements)  
- [⚙️ Installation](#️-installation)  
- [▶️ Running the App](#️-running-the-app)  
- [🖼️ Screenshots](#️-screenshots)  
- [📥 Optional: Download Kaggle Dataset](#-optional-download-kaggle-dataset)  
- [🔄 Workflow Pipeline](#-workflow-pipeline)  
- [🛠️ Customization Ideas](#️-customization-ideas)  
- [🐞 Troubleshooting](#-troubleshooting)  
- [📜 License](#-license)  
- [🙏 Credits](#-credits)

---

# ✨ Overview

This project provides a **local playground** for exploring two classic deep-learning colorization models:

- **ECCV16** (Classification-based 313-color bins)  
- **SIGGRAPH17** (Colorization with local + global hints)

Both models are provided as **pretrained PyTorch checkpoints**, and the Streamlit UI makes the process easy:

✔ Upload grayscale or RGB image  
✔ Auto-convert to LAB  
✔ Run both models  
✔ View side-by-side results  
✔ Download colorized outputs  
✔ Inspect dominant colors  

---

# 🎯 Goal & Problem Definition

### **🎯 Goal**
To offer an *offline*, easy-to-use UI for experimenting with deep image colorization models.

### **🧠 Problem Definition**
Given an image in grayscale (or RGB → Luminance), infer plausible **chrominance channels (`a/b`)** using CNNs trained on millions of natural images.  
This repo focuses specifically on **inference**, not training.

---

# 📂 Repository Structure

```
Gray to RGB/
├── app.py                                     # Streamlit app that loads both PyTorch models
├── ECCV16_and_SIGGRAPH17_Colorization.ipynb    # Exploratory notebook (same workflow as app)
├── ECCV16_and_SIGGRAPH17_Colorization.ipynb - Colab.pdf  # Notebook export for sharing
├── README.md                                 # This document
├── colorization_release_v2-9b330a0b.pth      # ECCV16 checkpoint (pretrained)
├── siggraph17-df00044c.pth                   # SIGGRAPH17 checkpoint (pretrained)
├── pts_in_hull.npy                           # AB color prior (from original repo)
├── images.jpg / grayscale-image-api.png / WhatsApp*.jpg  # Misc reference imagery
├── landscape_Images/
│   ├── gray/                                 # (optional) grayscale samples (currently empty)
│   └── color/                                # (optional) reference RGB samples
├── tfenv/                                    # Standalone Python environment with TensorFlow (optional)
└── .venv/                                    # Optional virtual environment for this project
```


> **Note:** Model weights must remain in the project root.

---

# ⭐ Features

- 🔌 **Works completely offline**  
- 🧠 Loads official **ECCV16 & SIGGRAPH17 PyTorch models**  
- 🧾 Fixes checkpoints with or without `module.` prefixes  
- 🎛️ Clean, modern **Streamlit UI**  
- 🎨 Side-by-side colorization preview  
- 💾 Download each model’s prediction as PNG  
- 🟦 LAB pre/post-processing included  
- 🎯 Dominant color palette extraction  
- 🚀 CPU/GPU auto-detection  

---

# 🧰 Tech Stack

| Layer | Tools |
|------|-------|
| UI / Serving | **Streamlit** |
| Models | **PyTorch 2.x**, ECCV16, SIGGRAPH17 |
| Imaging Tools | Pillow, scikit-image |
| Utilities | NumPy, KaggleHub (optional) |
| Environment | Python 3.9+, virtualenv/venv, optional CUDA |

---

# 📦 Requirements

- Python **3.9+**
- PyTorch **2.x**
- GPU optional (PyTorch auto-detects CUDA)

### Required packages:
```
torch
torchvision
streamlit
numpy
scikit-image
pillow
```

---

# ⚙️ Installation

### 1️⃣ Clone the repo

```bash
git clone https://github.com/yourusername/Gray-to-RGB.git
cd Gray-to-RGB
