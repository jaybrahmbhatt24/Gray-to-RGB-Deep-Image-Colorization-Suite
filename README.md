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
```

### 2️⃣ Install dependencies

For CUDA 12.1:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install streamlit numpy scikit-image pillow
```

CPU-only:
```bash
pip install torch torchvision torchaudio
```

Optional TensorFlow environment:
```bash
source tfenv/bin/activate
```
### ▶️ Running the App

Ensure the following files are in the root:

- colorization_release_v2-9b330a0b.pth

- siggraph17-df00044c.pth

Then run:
```bash
streamlit run app.py
```

The app opens at:
```bash
http://localhost:8501
```

Upload an image → view results → download PNGs.

### 🖼️ Screenshots
#### UI Preview



<img width="1919" height="1079" alt="Screenshot 2025-11-20 194409" src="https://github.com/user-attachments/assets/f720acbf-d665-4498-b2e3-7353ec6ba597" />
<img width="1919" height="1079" alt="Screenshot 2025-11-20 194511" src="https://github.com/user-attachments/assets/c6083f35-95bd-4c77-a9e5-32d3629b7dd1" />
<img width="700" height="466" alt="tiger" src="https://github.com/user-attachments/assets/f42557ad-349e-42eb-9ab7-de7aab74ca88" />
<img width="1919" height="1079" alt="Screenshot 2025-11-20 194525" src="https://github.com/user-attachments/assets/387fe5cf-7d37-43ab-8f5a-0e03ae13c2e8" />
<img width="700" height="466" alt="grayscale-image-api" src="https://github.com/user-attachments/assets/e63aaea5-fa26-40a9-bdf7-77e226444d45" />
<img width="1919" height="1079" alt="Screenshot 2025-11-20 194604" src="https://github.com/user-attachments/assets/76bd6f80-5ac7-4542-af38-228438b1a538" />



### 📥 Optional: Download Kaggle Dataset

Use KaggleHub to download the Landscape Colorization dataset:
```bash
import kagglehub

path = kagglehub.dataset_download("theblackmamba31/landscape-image-colorization")
print("Dataset downloaded to:", path)
```

### 🔄 Workflow Pipeline

- Upload grayscale or RGB image
- Convert to LAB
- Extract L channel
- Feed L into both models
- ECCV16 → 313-color probability bins
- SIGGRAPH17 → regression + bins
- Reconstruct LAB → RGB
- Display colorized images
- Extract dominant color palette
- Download output PNGs

### 🛠️ Customization Ideas

- Add color intensity/saturation sliders
- Add histogram/palette visualization
- Batch processing mode
- REST API version (FastAPI/Flask)
- Add TensorFlow colorization models
- Add drag-to-paint hints (like original SIGGRAPH17 demo)

### 🐞 Troubleshooting

| Issue               | Solution                                |
| ------------------- | --------------------------------------- |
| `FileNotFoundError` | Ensure `.pth` files are in project root |
| CUDA not detected   | Install correct CUDA PyTorch wheel      |
| Browser won’t open  | Manually visit `http://localhost:8501`  |
| state_dict mismatch | App auto-fixes `module.` prefixes       |

### 📜 License

Model weights follow the original license from the authors:
🔗 [https://github.com/richzhang/colorization](https://github.com/richzhang/colorization)

All additional code is MIT-licensed unless specified.

### 🙏 Credits

ECCV16 & SIGGRAPH17 models created by:

Richard Zhang, Phillip Isola, Alexei A. Efros
🔗 [https://github.com/richzhang/colorization](https://github.com/richzhang/colorization)

UI, pipeline design, and implementation by Jay Brahmbhatt.

### ⭐ Support the Project

If this project helps you:

- ⭐ Star the repo
- 🔄 Share it
- 🛠️ Contribute