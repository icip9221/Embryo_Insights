Thank you! Based on your actual project structure and architectural pattern (Chain of Responsibility + Builder), here is a tailored `README.md` for your repository `Embryo_Insights`:

---

# From Images to Insights: Cell Counting and Uniformity Grading of Day 3 Embryos

This repository contains the official implementation of our paper:

> **From Images to Insights: Cell Counting and Uniformity Grading of Day 3 Embryos**
> Nguyen Duy Tan, Tran Phuong Huy, Tran Thi Thanh Thuy, Hoang Thi Diem Tuyet, Dang Truong Son, Pham The Bao, Vu Ngoc Thanh Sang

## 🧬 Overview

This project presents a robust and automated pipeline for evaluating Day 3 embryos by performing **blastomere detection**, **boundary refinement**, and **uniformity assessment**. The approach integrates deep learning (YOLOv8) with advanced image processing techniques (GVF-based active contours) and calculates a novel metric, **Normalized Uniformity Value (NUV)**, for grading embryo quality.

The system is designed using the **Chain of Responsibility** and **Builder** design patterns, ensuring modularity, flexibility, and easy extension for different embryo analysis pipelines.

---

## 📁 Project Structure

```
Embryo_Insights/
├── main.py                         # Entry point to run the full embryo analysis pipeline
├── requirements.txt                # Python dependencies
├── LICENSE                         # License information
├── README.md                       # Project description (you are here)
│
├── abstract/                       # Abstract base classes for algorithm handlers and processes
├── adapter/                        # YOLOv8 detection adapter and utility functions
├── algorithms/                     # Implementations of cell enhancement, detection, segmentation, uniformity
├── config/                         # YAML config files for algorithm chain setups
├── manager/                        # Pipeline manager to run the defined process chain
├── registry/                       # Builder & registry for constructing pipeline based on configs
├── data/                           # Sample embryo image datasets (microscope/timelapse)
├── scripts/                        # Shell scripts to run the pipeline for different datasets
└── weights/                        # Pretrained detection models (YOLOv8, YOLOv11, YOLOX, etc.)
```

---

## ⚙️ How It Works

### 🔗 Chain of Responsibility + Builder Pattern

Each analysis stage (detection → contour refinement → uniformity grading) is implemented as a separate algorithm module, connected dynamically through a configurable pipeline:

* `abstract/`: Defines `abs_algorithm_process_handler.py` and related interfaces.
* `registry/`: Dynamically builds processing chains based on config files using `process_builder.py`.

### 🧪 Workflow

1. **Blastomere Detection**:
   YOLOv8 detects cell locations and provides bounding ellipses.

2. **Contour Refinement**:
   GVF-based snake model refines detected boundaries, improving precision.

3. **Uniformity Grading**:
   NUV is calculated from refined masks to evaluate cell size consistency.

---

## 🚀 Getting Started

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ Run the Pipeline

```bash
python main.py --pipeline_config config/GVFSnake.yaml --image data/microscope/xcells.jpg
```

### 🖥️ Batch Processing with Shell Scripts

Before running the provided shell scripts, make sure they are executable:

```bash
chmod +x scripts/microscope_script.sh
chmod +x scripts/timelapse_script.sh
```

Then run them:

```bash
./scripts/microscope_script.sh
./scripts/timelapse_script.sh
```

---

### 🔧 Configurable Pipelines

You can customize or switch processing pipelines via YAML files in the `config/` directory:

* `GVFSnake.yaml`: Uses GVF-snake for refinement
* `Snake.yaml`: Basic snake contour refinement

---

## 📊 Results & Contributions

### ✅ Main Contributions

1. **Hybrid architecture** combining YOLOv8 and GVF-snake for precise grading.
2. **Reduced subjectivity** in embryo evaluation via automation.
3. **NUV metric** introduced to objectively assess uniformity across cells.

### 🧪 Evaluation

The model shows high robustness in:

* Overlapping cell detection
* Low-contrast environments
* Accurate spatial boundary estimation

---

1. **A new section** acknowledging the reimplementation of the ESAVA method for comparison.
2. **Citation** of the original ESAVA paper using both a sentence and a formal BibTeX entry.

---

## 🧪 Comparative Benchmarking

To evaluate the effectiveness of our proposed method, we **reimplemented the ESAVA method** from the original repository:

🔗 [https://github.com/fsccycy/ESAVA](https://github.com/fsccycy/ESAVA)

This model, originally developed by Liao et al. (2024), provides an established baseline for blastomere contour refinement and embryo evaluation. We used it as a benchmark to **compare segmentation accuracy, grading consistency, and robustness** against our hybrid YOLOv8 + GVF-Snake approach.

> 📖 **Citation for ESAVA:**

```
Liao Z, Yan C, Wang J, Zhang N, Yang H, Lin C, Zhang H, Wang W, Li W.  
A clinical consensus-compliant deep learning approach to quantitatively evaluate human in vitro fertilization early embryonic development with optical microscope images.  
*Artificial Intelligence in Medicine*. 2024 Mar;149:102773.  
doi: 10.1016/j.artmed.2024.102773  
PMID: 38462274.
```
---


## 📚 Citation

```bibtex
@inproceedings{nguyen2025embryo,
  title={From Images to Insights: Cell Counting and Uniformity Grading of Day 3 Embryos},
  author={Nguyen, Duy Tan and Tran, Phuong Huy and Tran, Thi Thanh Thuy and Hoang, Thi Diem Tuyet and Dang, Truong Son and Pham, The Bao and Vu, Ngoc Thanh Sang},
  year={2025}
}
```

---

## 🙏 Acknowledgments

* IC-IP Lab, Saigon University
* Hung Vuong Hospital
* HP Fertility, Hai Phong International Hospital

---

## 📬 Contact

📧 **[vungocthanhsang@sgu.edu.vn](mailto:vungocthanhsang@sgu.edu.vn)**

Let me know if you'd like help generating example config files, training the detection model, or packaging the project for PyPI or Docker.
