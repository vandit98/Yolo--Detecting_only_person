# Computer Vision Tasks Overview

The following tasks cover a diverse set of computer vision applications, each presenting unique challenges and their respective solutions.

## Task I: Object Detection with YOLOv8

### Objective
Implement YOLOv8 for person detection, fine-tune on a custom dataset with at least 4 augmentations.

### Steps
1. Use a pre-trained YOLOv8 model.
2. Create a custom dataset and apply augmentations.
3. Train the model, save weights.
4. Perform inference on 4 videos, saving the output.

### Challenges and Solutions
- **Challenges:** Custom dataset creation, augmentation selection.
- **Solutions:** Rigorous documentation, trial-and-error for augmentation choices.

## Task II: Face Detection and Emotion Classification

### Objective
Train YOLOv8 for face detection, then classify emotions (Happy, Sad, Neutral) using a provided dataset.

### Steps
1. Use the provided face mask detection dataset.
2. Train YOLOv8 for face detection.
3. Crop faces and classify emotions.
4. Display emotion class on bounding box in 4 videos.

### Challenges and Solutions
- **Challenges:** Integration of two models, emotion classification.
- **Solutions:** Clear dataset understanding, modular code design.

## Task III: Image Super-resolution

### Objective
Utilize pre-trained models (SRGAN, ESRGAN, SWINFIR) for single-image super-resolution.

### Steps
1. Experiment with three super-resolution architectures.
2. Compare image quality with various upscaling factors.
3. Save source images, upscaled images, and documentation.

### Challenges and Solutions
- **Challenges:** Model selection, upscaling factor experimentation.
- **Solutions:** Systematic comparison, documentation for findings.

## Task IV: Vehicle Counting and Speed Estimation

### Objective
Use YOLOv8 for vehicle detection, count vehicles, and estimate their speed.

### Steps
1. Identify vehicles using YOLOv8.
2. Count vehicles and estimate speed in the video.
3. Save source video, inference code, inference video, and documentation.

### Challenges and Solutions
- **Challenges:** Speed estimation accuracy, vehicle tracking.
- **Solutions:** Thorough testing, possibly integrating additional tracking methods.

In all tasks, clear documentation, systematic experimentation, and modular code design are essential. The provided zip files for each task should contain necessary datasets, weight files, source videos, inference codes, and documentation for each task's process and outcomes.
