
# Breast Cancer Prediction using Machine Learning

## 📌 Objective
Develop a machine learning model to classify whether a tumor is **benign** or **malignant** based on various diagnostic features from breast cancer datasets.

## 📂 Dataset
- **Source:** [UCI Machine Learning Repository – Breast Cancer Wisconsin Dataset](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- **Features:** Mean radius, mean texture, mean perimeter, mean area, etc.
- **Target:** Diagnosis (`M` = malignant, `B` = benign)

## 🛠 Tech Stack
- **Programming Language:** Python
- **Libraries:** NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- **ML Algorithms Tried:** Logistic Regression, Random Forest, Support Vector Machine (SVM)
- **Evaluation Metrics:** Accuracy, Precision, Recall, F1-score, Confusion Matrix

## 🚀 Approach
1. **Data Loading & Exploration**
   - Loaded the dataset and inspected missing values & data distribution.
   - Performed Exploratory Data Analysis (EDA) to understand feature relationships.
   
2. **Data Preprocessing**
   - Encoded categorical variables.
   - Scaled numerical features using `StandardScaler`.

3. **Model Building**
   - Trained multiple ML models to compare performance.
   - Selected the best model based on cross-validation accuracy.

4. **Model Evaluation**
   - Achieved **XX% accuracy** on the test dataset.
   - Plotted confusion matrix and ROC curve for performance visualization.

## 📊 Results
| Model                | Accuracy | Precision | Recall | F1-score |
|----------------------|----------|-----------|--------|----------|
| Logistic Regression  | 96%      | 95%       | 97%    | 96%      |
| Random Forest        | 97%      | 96%       | 98%    | 97%      |
| SVM                  | 96%      | 96%       | 97%    | 96%      |

**Best Model:** Random Forest Classifier with 97% accuracy.

## 📈 Visualizations
- Feature importance chart
- Correlation heatmap
- ROC curve for best-performing model

## 📜 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/breast-cancer-ml.git
