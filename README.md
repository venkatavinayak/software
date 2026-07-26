# Software Defect Prediction Pipeline

An optimized, end-to-end, and completely leakage-free machine learning workflow designed to predict software defects using the NASA JM1 dataset. 

This project demonstrates software engineering best practices for building robust and reproducible machine learning code, featuring scikit-learn `Pipeline` architectures, cross-validated hyperparameter tuning, out-of-fold decision threshold optimization, ensembling, and explainability using SHAP.

## Project Structure

*   **`SoftwareEngineer (1).ipynb`**: An interactive, step-by-step Jupyter Notebook containing exploratory data analysis (EDA), preprocessing, training, threshold tuning, and model interpretation.
*   **`pipeline.py`**: The production-ready script to run the end-to-end pipeline from the command line.
*   **`utils.py`**: A helper module containing functions for data loading, preprocessing pipelines, model parameters, and metrics calculation.
*   **`outputs/`**: Folder containing the exported models and reports:
    *   `best_model.pkl`: The final trained model pipeline (includes preprocessing and classifier).
    *   `preprocessing_pipeline.pkl`: The fitted preprocessing steps.
    *   `best_threshold.json`: Configured optimal threshold values for each classifier.
    *   `metrics.csv`: Performance metrics for all models (before and after threshold tuning).
    *   `feature_importance.csv`: Ranked list of features based on SHAP values.
    *   `model_comparison.png`: Accuracy, F1-Score, and ROC-AUC comparisons.
    *   `threshold_impact_f1.png`: Visualization of F1-Score improvements before and after decision threshold tuning.
    *   `roc_curves.png` & `precision_recall_curves.png`: Performance curves for top models.
    *   `shap_summary.png`: SHAP feature importance plot.

## Core Features & Methodology

1.  **Strict Anti-Leakage Design:** All preprocessing steps (median imputation, outlier Winsorization, and scaling) are placed inside scikit-learn `Pipeline` objects. Preprocessing fits only on the training folds during cross-validation, eliminating data leakage.
2.  **McCabe & Halstead Normalization:** Count-based software metrics (e.g. lines of code, volume, difficulty) are heavily right-skewed. The pipeline applies a `log1p` transformation followed by a `RobustScaler` to ensure linear, distance-based, and probabilistic models perform optimally.
3.  **Handling Class Imbalance:**
    *   Uses cost-sensitive training (`class_weight='balanced'` and `scale_pos_weight`) to force base classifiers to pay attention to the minority defect class (~19% positive).
    *   Performs **Out-of-Fold Decision Threshold Tuning**: Uses `cross_val_predict(method='predict_proba')` to obtain out-of-fold validation probabilities, and searches thresholds from `0.10` to `0.90` (increments of `0.02`) to find the threshold that maximizes the F1-score.
4.  **Tuned Base Classifiers:** Hyperparameters are tuned using `RandomizedSearchCV` with Stratified 5-Fold Cross-Validation for:
    *   Logistic Regression (`C`)
    *   Support Vector Machine (`C`, `gamma`)
    *   K-Nearest Neighbors (`n_neighbors`)
    *   Decision Tree (`max_depth`)
    *   Random Forest (`n_estimators`, `max_depth`, `min_samples_split`)
    *   XGBoost (`n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `gamma`)
5.  **Ensemble Architectures:** Trains and compares three ensembling strategies:
    *   **Hard Voting Ensemble**: Majority vote of top classifiers.
    *   **Soft Voting Ensemble**: Average of predicted probabilities.
    *   **Stacking Ensemble**: Stacks predictions of base classifiers using a Logistic Regression meta-classifier.
6.  **Interpretability (SHAP):** Leverages `shap.TreeExplainer` on the best tree-based model to extract fast and exact Shapley feature importances, showing which software metrics contribute most to defect risk.

## Installation & Setup

1.  Clone this repository:
    ```bash
    git clone https://github.com/venkatavinayak/software.git
    cd software
    ```
2.  Install dependencies:
    ```bash
    pip install numpy pandas scikit-learn xgboost matplotlib shap joblib
    ```

## Usage

### Run from the Command Line
Run the end-to-end training and evaluation:
```bash
python pipeline.py
