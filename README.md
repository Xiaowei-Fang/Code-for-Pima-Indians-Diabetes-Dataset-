Diabetes Prediction with SVC and Extra Trees Classifier

This repository contains the Python Jupyter Notebook: diabetes_prediction.ipynb

## Project Goal:
This notebook implements and evaluates Support Vector Classifier (SVC) and Extra Trees Classifier for predicting diabetes using the Pima Indians Diabetes Dataset. It focuses on robust data preprocessing, including handling missing values, feature scaling, and using SMOTE (Synthetic Minority Over-sampling Technique) correctly on the training data only to prevent data leakage.

## Dataset:
Pima Indians Diabetes Dataset (expected as diabetes.csv).

## Key Dependencies

The notebook relies on common Python libraries for data science and machine learning:
*   `pandas` for data manipulation
*   `numpy` for numerical operations
*   `matplotlib` and `seaborn` for visualization (though visualization outputs are not the primary focus of the final model evaluation in the notebook)
*   `scikit-learn` for:
    *   Preprocessing (`RobustScaler`, `StandardScaler`, `MinMaxScaler`)
    *   Model selection (`train_test_split`, `RandomizedSearchCV`)
    *   Classifiers (`SVC`, `ExtraTreesClassifier`)
    *   Metrics (`accuracy_score`, `confusion_matrix`, `classification_report`)
*   `imblearn` for `SMOTE` (oversampling)

## Results Summary

After preprocessing, hyperparameter tuning, and evaluation on the test set (20% of the data):
*   **Support Vector Classifier (SVC):** Achieved an accuracy of **80.5%**.
*   **Extra Trees Classifier:** Achieved an accuracy of **84.0%**.

Detailed confusion matrices and classification reports are available within the notebook.


## Note

This implementation prioritizes robust validation practices, such as applying SMOTE exclusively to the training data. 
The accuracies achieved reflect this approach and may differ from studies employing alternative preprocessing or validation strategies. 
The primary aim is to provide a transparent and methodologically sound baseline for these classifiers on this dataset.
