# Classic Computer Vision – Image Basics

This repository covers the fundamentals of **Computer Vision** using classical techniques (before deep learning).

The goal is to understand how images are represented and processed as numerical data.

---

## What This Project Covers

- Loading images using OpenCV
- Understanding image shapes and channels
- RGB vs Grayscale images
- Image resizing for machine learning models
- Basic image preprocessing

---

## Key Concepts Learned

- Images are matrices of numbers, not just pictures
- Color images have 3 channels (RGB)
- Grayscale images reduce complexity
- ML and CV models require fixed-size inputs
- Preprocessing is a critical step before CNNs

---

## Image Processing Techniques
- Gaussian blurring to reduce noise
- Canny edge detection for object boundaries
- Sobel filters to detect directional edges

## Key Learnings
- Noise affects edge detection
- Blurring improves edge quality
- Edges represent important visual features

---

## Feature Extraction with HOG

- Implemented Histogram of Oriented Gradients (HOG)
- Used SVM for digit classification
- Demonstrates classic CV + ML pipeline before CNNs

## Why This Matters
- Shows how features represent image structure
- Helps understand how CNNs learn similar patterns automatically

---

## Project Structure

classic-computer-vision/
├── images/
│ └── sample.jpg
├── load_image.py
├── preprocess.py
├── blur.py
├── edges.py
├── sobel.py
├── hog_svm_digits.py
├── README.md

## What I Learned
- Images are numerical matrices
- RGB vs Grayscale images
- Image shape and channels
- Image resizing for ML models

## Tools Used
- Python
- OpenCV
- Matplotlib
