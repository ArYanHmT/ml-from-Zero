# 05 - Spam Email Classifier Evaluation 📊

This project focuses on evaluating a Machine Learning model for spam email detection.

## 🎯 Project Goal
Understand how to measure model performance using metrics like:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## 🧠 What I Learned
- Train/Test split for evaluation
- Calculating performance metrics
- Understanding model mistakes
- Confusion matrix visualization
- Why Accuracy alone is not enough

## 🛠 Technologies Used
- Python
- Pandas
- Scikit-learn

## 🚀 How It Works
1. Load spam email dataset
2. Split data into training and testing sets
3. Convert text to numeric features using TF-IDF
4. Train Logistic Regression model
5. Predict emails in test set
6. Calculate Accuracy, Precision, Recall, F1 Score
7. Build Confusion Matrix to analyze errors

## 📁 Project Structure
05_model_evaluation/ │── model_evaluation.py │── README.md
Copy code

## 📈 Result
Metrics now reflect model performance properly, showing which emails are correctly or incorrectly classified.

## 🔮 Future Improvements
- Use a larger and real spam dataset
- Try different models
- Visualize confusion matrix as a heatmap