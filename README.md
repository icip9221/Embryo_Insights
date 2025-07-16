# 🧬 From Images to Insights: Cell Counting and Uniformity Grading of Day 3 Embryos

This repository contains the official implementation of our paper:

> **From Images to Insights: Cell Counting and Uniformity Grading of Day 3 Embryos**
> Nguyen Duy Tan, Tran Phuong Huy, Tran Thi Thanh Thuy, Hoang Thi Diem Tuyet, Dang Truong Son, Pham The Bao, Vu Ngoc Thanh Sang

---

## 📌 Overview

This project presents a robust and automated pipeline for evaluating **Day 3 embryos** by performing **blastomere detection**, **boundary refinement**, and **uniformity assessment**.
Our hybrid approach integrates:

* **YOLOv8-based object detection**
* **GVF-based active contour refinement**
* A novel **Normalized Uniformity Value (NUV)** for grading consistency

The architecture follows the **Chain of Responsibility** and **Builder** design patterns for clean modularity and extensibility.

---

## 📁 Project Structure

```
Embryo_Insights/
├── main.py                         # Entry point for executing the full pipeline
├── requirements.txt                # Python dependencies
├── LICENSE                         # License information
├── README.md                       # You are here
│
├── abstract/                       # Abstract base classes for all processes
├── adapter/                        # YOLOv8 ONNX adapter and inference tools
├── algorithms/                     # Core implementations (enhancement, detection, contours, grading)
├── config/                         # Pipeline configuration files (YAML)
├── manager/                        # Chain-of-responsibility handler
├── registry/                       # Registry and builder for pipeline creation
├── data/                           # Sample microscope and timelapse embryo images
├── scripts/                        # Scripts for batch evaluation
└── weights/                        # Directory for pretrained detection models
```

---

## ⚙️ How It Works

### 🔗 Modular Pipeline with Chain of Responsibility

The processing stages are implemented as standalone modules:

* Detection (`YOLOv8`)
* Segmentation (`GVF Snake`, `ESAVA`, or `TV/Bilateral`)
* Uniformity scoring (`NUV`)

Each component is dynamically registered and built using YAML configs and Python classes.

### 🔁 Workflow

1. **Detection** – YOLOv8 locates blastomeres using bounding ellipses.
2. **Refinement** – Active contour or ESAVA segmentation refines the boundary.
3. **Grading** – Calculates **NUV** from areas to assess cell uniformity.

---

## 🚀 Getting Started

### 📦 Install Dependencies

```bash
# Create virtual environment (Linux/macOS)
python3 -m venv env

# Activate the environment
source env/bin/activate

# --- OR on Windows ---
# python -m venv env
# .\env\Scripts\activate

```

```bash
pip install -r requirements.txt
```

### 📥 Download Pretrained Weights

Download the YOLOv8 object detection model weights here:

🔗 **[Download YOLOv8 Weights](https://drive.google.com/drive/folders/1suOTlOYGPH2i7BkMhlE92GpJN3rzUh6y?usp=sharing)**

Then place them inside the `weights/` directory.

> These weights are trained specifically on Day 3 embryo microscope and timelapse data.

---

### ▶️ Run the Pipeline on a Single Image

```bash
python main.py --pipeline_config config/GVFSnake.yaml --image data/microscope/xcells.jpg
```

---

### 📂 Run on Multiple Images with Scripts

Before running:

```bash
chmod +x scripts/microscope_script.sh
chmod +x scripts/timelapse_script.sh
```

Then execute:

```bash
./scripts/microscope_script.sh
./scripts/timelapse_script.sh
```

---

### ⚙️ Configurable Pipeline via YAML

Switch between different processing methods via YAML in `config/`:

* `GVFSnake.yaml`: Uses GVF-based snake model
* `Snake.yaml`: Uses basic active contour segmentation
* `ESAVA.yaml`: Uses the reimplemented ESAVA method

---

## 📊 Main Contributions

* ⚙️ A hybrid framework combining YOLOv8 detection with GVF-snake contour refinement
* 🧠 Introduction of the **NUV (Normalized Uniformity Value)** for objective blastomere grading
* 📦 Modular design using **Chain of Responsibility** + **Builder pattern**
* 🤖 Robust performance on microscope and time-lapse embryo images
* 🔁 Built-in support for comparing alternative segmentation methods like **ESAVA**

---

## 🧪 Comparative Benchmarking

We reimplemented the **ESAVA method** from the official repository to provide a comparison baseline for segmentation and grading:

🔗 [https://github.com/fsccycy/ESAVA](https://github.com/fsccycy/ESAVA)

This reimplementation was evaluated against our proposed GVF-enhanced contouring pipeline to assess accuracy and grading consistency.

> 📖 **Citation for ESAVA**:

```
Liao Z, Yan C, Wang J, Zhang N, Yang H, Lin C, Zhang H, Wang W, Li W.  
A clinical consensus-compliant deep learning approach to quantitatively evaluate human in vitro fertilization early embryonic development with optical microscope images.  
Artificial Intelligence in Medicine. 2024 Mar;149:102773.  
doi: 10.1016/j.artmed.2024.102773  
PMID: 38462274.
```

## 🙏 Acknowledgments

* 🏥 Hung Vuong Hospital, Ho Chi Minh city, Vietnam
* 🧪 HP Fertility Center – Hai Phong International Hospital, Hai Phong city, Vietnam
* 💻 IC-IP Lab, Faculty of Information Technology, Saigon University, Ho Chi Minh city, Vietnam

---

## 📬 Contact

📧 **[vungocthanhsang@sgu.edu.vn](mailto:vungocthanhsang@sgu.edu.vn)**
📧 **[ngduytan288@gmail.com](mailto:ngduytan288@gmail.com)**

---
