# White Wine Quality Prediction Using Machine Learning

This project aims to predict the quality of white wine using various machine learning algorithms based on physicochemical characteristics such as acidity, residual sugar, alcohol, and sulfur dioxide levels.

## Dataset

- Source: UCI Machine Learning Repository  
- Number of instances: 4898  
- Features: 11 chemical attributes + 1 target (quality)
-We created the test set using stratified sampling to maintain proportional representation of
each class. The dataset had no missing values, but 937 duplicate rows were found and removed.
We applied standard normalization (z-score) using StandardScaler to scale numerical
features to have zero mean and unit variance. This step is essential especially for models
like SVM and KNN which are sensitive to feature scales. The only categorical feature, ”wine
color”, was encoded using one-hot encoding.

## Problem Statement

Wine quality is often evaluated by human tasters, which can be subjective and time-consuming. By using machine learning models, we aim to provide a faster, more consistent, and objective prediction of wine quality.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas, NumPy
- scikit-learn, imbalanced-learn, XGBoost
- Seaborn & Matplotlib for visualization

## Models Tested

The selected models were chosen to cover a range of algorithmic strategies: linear (Logistic
Regression), distance-based (KNN), kernel-based (SVM), and ensemble methods (Random
Forest, Gradient Boosting, XGBoost). This diversity allows for a robust comparison.
Among all the tested models, Random Forest performed best in terms of weighted F1-
score (0.609) and ROC-AUC (0.856) on the test set. It also achieved a strong balance
between precision and recall for both classes. We trained the following models using 5-fold
cross-validation:
• Logistic Regression
• K-Nearest Neighbors (KNN)- (k=5, uniform; k=10, distance; k=21, distance)
• Support Vector Machine (SVC)
• Random Forest
• Gradient Boosting
• XGBoost

## Final Test Performance
-To complement the cross-validation results, we also evaluated all models on the held-out test
set. The test results show that Random Forest achieved the best overall performance, with
an F1-score of 0.609 and a ROC-AUC of 0.856. These metrics indicate a strong balance
between precision and recall, particularly for the minority class. Other ensemble models
like Gradient Boosting and XGBoost also performed competitively, but their F1-scores were
slightly lower. Linear models such as Logistic Regression showed high recall but suffered
from low precision, leading to lower F1-scores. KNN models had moderate success, especially
when using distance weighting, but underperformed compared to ensemble methods. These
findings suggest that tree-based ensemble models, especially Random Forest, are well-suited
for this classification task due to their ability to capture non-linear relationships and handle
imbalanced datasets effectively.

-The final model, a fine-tuned Random Forest classifier, achieved a test F1-score of 0.609
and ROC-AUC of 0.856, indicating solid performance on an imbalanced classification task.
However, the performance on the minority class was still lower than that of the majority
class, suggesting that class imbalance and overlapping feature distributions pose challenges.

## Files

- `winequality.ipynb`: Main Jupyter Notebook with full analysis and model training
- `README.md`: Project overview 
- `Wine_Quality-Explainable AI(XAI).ipynb`:  Jupyter Notebook performing explainability analysis with SHAP, PDPs, and feature importance.
- `Explainability_Report.pdf`: Detailed report on model explainability using SHAP, Permutation Importance, and Partial Dependence Plots (PDP).
