# 🧠 Brain Tumor Detection using Basic CNN

A Deep Learning project that utilizes a **Convolutional Neural Network (CNN)** to detect brain tumors from 2D MRI scans. Built from scratch using **PyTorch**.

## 📌 Project Overview
This project classifies Brain MRI images into two categories:
*   **Yes:** The MRI shows a Brain Tumor.
*   **No:** The MRI is Healthy.

The model is trained on a binary classification dataset and runs efficiently on NVIDIA GPUs.

## 🚀 Features
*   **Custom CNN Architecture:** Built from scratch using `torch.nn` (3 Convolutional Layers).
*   **GPU Acceleration:** Optimized for **NVIDIA RTX 2050** (CUDA Support).
*   **Data Augmentation:** Resizing, Normalization, and Tensor conversion.
*   **Inference Script:** Test the model on any custom image downloaded from the internet.
*   **High Accuracy:** Achieved ~94% accuracy on the test set.

## 🛠️ Tech Stack
*   **Language:** Python 3.x
*   **Framework:** PyTorch
*   **Libraries:** Torchvision, PIL (Pillow), Scikit-Learn
*   **Hardware:** NVIDIA GeForce RTX 2050 (4GB VRAM)

## 📂 Project Structure
Ensure your folder is organized like this:

```text
BrainTumorProject/
│
├── Data/                   # Dataset Folder
│   ├── yes/                # Images containing tumors
│   └── no/                 # Healthy brain images
│
├── train.py                # Script to train the model
├── predict_custom.py       # Script to test on new images
├── brain_tumor_model.pth   # Saved model weights (generated after training)
└── README.md               # Project documentation

└── README.md               # Project documentation
⚙️ Installation & Setup
Clone or Download this repository.
Install Dependencies:
Open your terminal and run:
code
Bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install scikit-learn matplotlib pillow
(Note: The command above ensures you get the GPU-supported version of PyTorch).
Prepare the Dataset:
Download the "Brain MRI Images for Brain Tumor Detection" dataset (e.g., from Kaggle).
Extract it into a folder named Data.
Ensure inside Data you have two folders: yes and no.

🏃‍♂️ How to Run
1. Train the Model
To train the CNN from scratch, run:
code
Bash
python train.py
This will train for 10 Epochs.
It will automatically save the best model as brain_tumor_model.pth.
2. Test on Custom Images
To diagnose a specific image (e.g., one you downloaded from Google):
Save the image in the project folder (e.g., named test1.jpg).
Open predict_custom.py and change the line: image_name = "test1.jpg".
Run the script:
code
Bash
python predict_custom.py
The AI will output the Prediction (Yes/No) and the Confidence Score.
🧠 Model Architecture
The model is a Basic CNN consisting of:
Input Layer: 128x128 RGB Images.
Conv Block 1: 32 Filters (3x3), ReLU, MaxPool.
Conv Block 2: 64 Filters (3x3), ReLU, MaxPool.
Conv Block 3: 128 Filters (3x3), ReLU, MaxPool.
Fully Connected Layer: Flattens the 3D features and outputs binary classification probabilities.
🔮 Future Improvements
Implement ResNet-18 for better feature extraction.
Upgrade to 3D ResNet to handle volumetric NIfTI/DICOM data.
Deploy as a Web App using Streamlit.
👨‍💻 Author
Pullari Shiva Kumar Yadav

- Data Science Enthusiast