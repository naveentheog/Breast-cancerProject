# Breast Cancer Classification using Machine Learning

## 📌 Objective
Build a machine learning model to classify whether a tumor is **benign** or **malignant** using the Breast Cancer Wisconsin dataset.

## 📂 Dataset
- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Features:** Diagnostic measurements (mean radius, texture, perimeter, area, etc.)
- **Target:** `0` = malignant, `1` = benign

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Scikit-learn
- **Algorithm:** Logistic Regression
- **Evaluation Metric:** Accuracy Score

## 🚀 Approach
1. **Data Loading & Exploration**
   - Loaded dataset directly from Scikit-learn.
   - Checked shape, feature names, and class distribution.
   
2. **Data Preprocessing**
   - Converted data to Pandas DataFrame.
   - Split into training and test sets.

3. **Model Training**
   - Used Logistic Regression to train on training data.
   - Increased `max_iter` to ensure convergence.

4. **Model Evaluation**
   - Measured accuracy score on both training and test sets.

## 📊 Results
- **Training Accuracy:** ~XX%
- **Testing Accuracy:** ~XX%
- Model successfully classifies tumors with high accuracy.

## 📈 Example Output
```python
Accuracy on training data:  0.9494505494505494
Accuracy on test data:  0.9298245614035088
