# BLG-453E COMPUTER VISION Project

Overview
This repository contains the work for Project 4 of the BLG-453E (Computer Vision) course at Istanbul Technical University. The project focuses on implementing and analyzing various computer vision techniques, including optical flow analysis and transfer learning.

Table of Contents
Project Description
Repository Structure
Installation
Usage
Results
Contributing
License
Acknowledgments
Project Description
The objectives of this project are:

Optical Flow Analysis: Implementing methods to analyze motion in videos.
Transfer Learning: Adapting pre-trained models for specific tasks.
Data Analysis: Conducting comprehensive analyses on provided datasets.
Repository Structure
The repository is organized as follows:

plaintext
Copy
Edit
BLG-453E---Project-4/
├── frames/                 # Extracted frames from videos
├── test/                   # Test dataset
├── train/                  # Training dataset
├── val/                    # Validation dataset
├── BLG_453E_Fall2425_HW4.pdf  # Project description document
├── Lion.jpg                # Project logo
├── README.md               # Project README file
├── best_model.pth          # Best model weights
├── irem.ipynb              # Jupyter Notebook for the project
├── opticalflow.py          # Script for optical flow analysis
├── opticalflow_video.mp4   # Video demonstrating optical flow results
├── output_video.mp4        # Output video of the project
├── q1.py                   # Script for question 1
├── q3.py                   # Script for question 3
├── q4.py                   # Script for question 4
├── q4_analysis.py          # Analysis script for question 4
├── roboteatingkebab.png    # Sample image
└── transferlearning.py     # Script for transfer learning
Installation
To set up the project locally:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/iremtaze/BLG-453E---Project-4.git
cd BLG-453E---Project-4
Install the required packages:

Ensure you have Python 3.x installed. Then, install the necessary packages:

bash
Copy
Edit
pip install -r requirements.txt
Note: If requirements.txt is not available, manually install the required packages as specified in the scripts.

Usage
To run the various components of the project:

Optical Flow Analysis:

Execute the opticalflow.py script to perform optical flow analysis:

bash
Copy
Edit
python opticalflow.py
Transfer Learning:

Run the transferlearning.py script to perform transfer learning tasks:

bash
Copy
Edit
python transferlearning.py
Other Scripts:

For other functionalities, execute the respective scripts:

bash
Copy
Edit
python script_name.py
Replace script_name.py with the desired script's filename.

Results
The project outputs include:

Optical Flow Result Video: opticalflow_video.mp4
Model Output Video: output_video.mp4
Trained Model Weights: best_model.pth
Contributing
Contributions are welcome. If you have suggestions or improvements, please fork the repository and submit a pull request. For major changes, open an issue to discuss the proposed modifications.

License
This project is licensed under the MIT License.

