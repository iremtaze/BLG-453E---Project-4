# BLG-453E - Project 

![Project Logo](Lion.jpg)

## 📌 Introduction

This repository contains the coursework for **Project 4** in the **BLG-453E (Computer Vision)** course at **Istanbul Technical University**. The project focuses on implementing and analyzing key computer vision techniques, including **optical flow analysis** and **transfer learning**.

## 📂 Repository Structure

The repository is organized as follows:

```
BLG-453E---Project-4/
├── frames/                 # Extracted frames from videos
├── test/                   # Test dataset
├── train/                  # Training dataset
├── val/                    # Validation dataset
├── BLG_453E_Fall2425_HW4.pdf  # Project description document
├── Lion.jpg                # Project logo
├── README.md               # Project documentation
├── best_model.pth          # Trained model weights
├── irem.ipynb              # Jupyter Notebook for experiments
├── opticalflow.py          # Optical flow analysis script
├── opticalflow_video.mp4   # Optical flow results video
├── output_video.mp4        # Processed output video
├── q1.py                   # Solution for Question 1
├── q3.py                   # Solution for Question 3
├── q4.py                   # Solution for Question 4
├── q4_analysis.py          # Data analysis script for Question 4
├── roboteatingkebab.png    # Sample image
└── transferlearning.py     # Transfer learning script
```

## 🛠️ Installation

To set up and run the project locally, follow these steps:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/iremtaze/BLG-453E---Project-4.git
   cd BLG-453E---Project-4
   ```

2. **Install Dependencies:**
   Ensure you have **Python 3.x** installed, then install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   *If `requirements.txt` is missing, install dependencies manually as specified in the scripts.*

## 🚀 Usage

Run the following scripts for different functionalities:

### 🔹 Optical Flow Analysis
```bash
python opticalflow.py
```
This script extracts motion information from videos using optical flow techniques.

### 🔹 Transfer Learning
```bash
python transferlearning.py
```
Runs a pre-trained model adapted to a specific task using transfer learning.

### 🔹 Additional Scripts
To execute any other script:
```bash
python script_name.py
```
Replace `script_name.py` with the appropriate filename.

## 📊 Results & Outputs

The project produces the following results:

- **🎥 Optical Flow Analysis Video:** [`opticalflow_video.mp4`](opticalflow_video.mp4)
- **🎥 Model Output Video:** [`output_video.mp4`](output_video.mp4)
- **🔧 Trained Model Weights:** [`best_model.pth`](best_model.pth)

## 📚 Project Description

The objectives of this project are:

- **Optical Flow Analysis:** Implementing methods to analyze motion in videos.
- **Transfer Learning:** Adapting pre-trained models for specific tasks.
- **Data Analysis:** Conducting comprehensive analyses on provided datasets.

## 👨‍💼 Contributing

Contributions are welcome! If you have suggestions or improvements, please fork the repository and submit a pull request. For major changes, open an issue to discuss the proposed modifications.

## 💍 License

This project is licensed under the [MIT License](LICENSE).



