# Face Emotion Detection Project

This project involves training a YOLOv8 object detection model for face detection using a face mask detection dataset. Subsequently, an image classification model is trained to classify emotions on the detected faces.

## Dataset

The dataset used for training the YOLOv8 model can be found [here](https://www.kaggle.com/datasets/deepakat002/face-mask-detection-yolov5). For emotion classification, you need a separate dataset containing labeled face images with emotions ('Happy', 'Sad', 'Neutral').

## Requirements

- Python 3.x
- PyTorch
- OpenCV
- Other dependencies (listed in `requirements.txt`)

Install the dependencies using:

```bash
pip install -r requirements.txt


/
|-- yolov8/                   # YOLOv8 model and related files
|-- emotion_classification/   # Emotion classification model and related files
|-- data/                     # Dataset and preprocessing scripts
|-- videos/                   # Input videos for testing
|-- src/                      # Source code files
|-- README.md
|-- requirements.txt
```

## Usage
Setup Environment: I have used conda environment

Install the required dependencies.
Train YOLOv8 Model:

Follow the instructions in the yolov8/ directory to train the YOLOv8 model on the face mask detection dataset.
Face Cropping:

Use the trained YOLOv8 model to detect faces and crop the regions.
Train Emotion Classification Model:

Prepare an emotion classification dataset and train the model using the scripts in the emotion_classification/ directory.
Display Results on Videos:

Run the provided script to display emotion labels on faces in input videos.
