# 04 - Digit Recognizer 🧠🔢

This project is a beginner Computer Vision project that classifies handwritten digits (0–9) using Machine Learning.

## 🎯 Project Goal
Build a model that can recognize handwritten numbers from images.

## 🧠 What I Learned
- Working with image datasets
- Understanding how images are converted to numbers
- Train/Test split
- Training a KNN classification model
- Making predictions
- Measuring model accuracy

## 📊 Dataset
This project uses the built-in Digits dataset from Scikit-learn:
- 8x8 grayscale images
- Numbers from 0 to 9
- Each image is converted into numerical features

## 🛠 Technologies Used
- Python
- Scikit-learn
- Matplotlib

## 🚀 How It Works
1. Load the digits dataset
2. Split data into training and testing sets
3. Train a KNN model using fit()
4. Predict digits using predict()
5. Evaluate performance using accuracy score
6. Display a sample digit image

## 📁 Project Structure
04_digit_recognizer/ │── digit_recognizer.py │── README.md
Copy code

## 📈 Result
The model achieves around 97%–99% accuracy in recognizing handwritten digits.

## 🔮 Future Improvements
- Try different models (SVM, Neural Networks)
- Use larger image datasets
- Build a simple UI to draw digits and predict them