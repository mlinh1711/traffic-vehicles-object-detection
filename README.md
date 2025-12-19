
---

# Traffic Vehicles Object Detection

This project implements a **traffic vehicle object detection** system using deep learning techniques in Computer Vision. The objective is to detect and localize vehicles in traffic scenes from images and videos, supporting applications such as traffic monitoring, intelligent transportation systems, and urban traffic analysis.

The implementation is provided in the form of a Jupyter Notebook to ensure clarity, reproducibility, and ease of experimentation.

---

## Project Overview

Traffic surveillance plays a crucial role in modern smart city infrastructure. Automating vehicle detection from traffic footage helps reduce manual observation, improve analysis accuracy, and enable real-time traffic management.

In this project, a deep learning–based object detection approach is applied to identify vehicles in traffic scenes. The workflow includes loading the dataset, applying a pretrained object detection model, performing inference, and visualizing detection results directly on images and videos.

---

## Dataset

Due to GitHub file size limitations, the dataset is **not included** in this repository.

The dataset used in this project is publicly available on Kaggle:

**Traffic Vehicles Object Detection Dataset**  
https://www.kaggle.com/datasets/saumyapatel/traffic-vehicles-object-detection

### Dataset Usage

After downloading and extracting the dataset, store it locally following a structure similar to:

```

Traffic Dataset/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── videos/

```

The notebook assumes the dataset is stored locally and accessed via file paths defined within the code.

---

## Repository Structure

```

traffic-vehicles-object-detection/
│
├── traffic-vehicles-object-detection.ipynb
├── requirements.txt
├── .gitignore
└── README.md

````

### File Description

- **traffic-vehicles-object-detection.ipynb**  
  Main notebook containing the complete object detection workflow, including dataset loading, model inference, and result visualization.

- **requirements.txt**  
  Specifies the Python dependencies required to run the notebook.

- **.gitignore**  
  Prevents large datasets, videos, and generated artifacts from being pushed to GitHub.

---

## Requirements

The project is implemented using **Python 3.9 or later**.

Install the required dependencies using:

```bash
pip install -r requirements.txt
````

Main libraries used in this project include:

* numpy
* pandas
* ultralytics
* opencv-python
* matplotlib
* PyYAML

Note that the `ultralytics` package automatically installs PyTorch. For GPU acceleration, ensure that the appropriate PyTorch version compatible with your CUDA setup is installed.

---

## How to Run

1. Clone this repository:

```bash
git clone https://github.com/mlinh1711/traffic-vehicles-object-detection.git
cd traffic-vehicles-object-detection
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the dataset from Kaggle and extract it to a local directory.

4. Open the Jupyter Notebook:

```bash
jupyter notebook traffic-vehicles-object-detection.ipynb
```

5. Run the notebook cells sequentially to perform vehicle detection and visualize the results.

---

## Results

The notebook produces visual detection outputs where vehicles are highlighted with bounding boxes on traffic images and video frames. These visualizations allow qualitative evaluation of detection performance and demonstrate the effectiveness of deep learning–based object detection for traffic scenes.

---

## Notes

* Large datasets and video files are intentionally excluded from this repository to keep it lightweight and easy to clone.
* All experiments are reproducible using the provided notebook and the external dataset link.

---
