(
echo # FinWise Lending — Baseline Model Report
echo.
echo ## 1. Overview
echo.
echo This report establishes the baseline performance benchmark for the FinWise Lending credit risk scoring system.
echo.
echo The baseline model uses an XGBoost classifier to predict the `loan_status` target from applicant income, employment, loan, and credit-history features.
echo.
echo The baseline provides a reference point for future model improvements such as hyperparameter tuning, feature engineering, calibration, SHAP explainability, and fairness analysis.
echo.
echo ---
echo.
echo ## 2. Model Configuration
echo.
echo | Configuration | Value |
echo |---|---|
echo | Model | XGBoost Classifier |
echo | Task | Binary Classification |
echo | Test Size | 20%% |
echo | Random State | 42 |
echo | Evaluation Threshold | 0.50 |
echo.
echo The train/test split was stratified to preserve the target-class distribution between the training and testing datasets.
echo.
echo Preprocessing was fitted only on the training data to prevent data leakage.
echo.
echo ---
echo.
echo ## 3. Evaluation Benchmark
echo.
echo | Metric | Baseline Result |
echo |---|---:|
echo | Accuracy | **0.9110** |
echo | Precision | **0.7947** |
echo | Recall | **0.7997** |
echo | F1 Score | **0.7972** |
echo | ROC-AUC | **0.9438** |
echo | Gini Coefficient | **0.8876** |
echo.
echo ### Gini Acceptance Criterion
echo.
echo The project acceptance requirement is:
echo.
echo **Gini >= 0.45**
echo.
echo Baseline result:
echo.
echo **Gini = 0.8876**
echo.
echo ### Status: PASS
echo.
echo The baseline model exceeds the required Gini coefficient by a substantial margin.
echo.
echo ---
echo.
echo ## 4. Gini Calculation
echo.
echo The Gini coefficient is derived from ROC-AUC using:
echo.
echo `Gini = 2 × ROC-AUC − 1`
echo.
echo For this baseline model:
echo.
echo ```text
echo ROC-AUC = 0.9438
echo Gini = (2 × 0.9438) − 1
echo Gini = 0.8876
echo ```
echo.
echo ---
echo.
echo ## 5. Confusion Matrix
echo.
echo The baseline model produced the following confusion matrix:
echo.
echo ```text
echo [[4773  293]
echo  [ 284 1134]]
echo ```
echo.
echo | | Predicted 0 | Predicted 1 |
echo |---|---:|---:|
echo | Actual 0 | 4773 | 293 |
echo | Actual 1 | 284 | 1134 |
echo.
echo ---
echo.
echo ## 6. Classification Report
echo.
echo ```text
echo               precision    recall  f1-score   support
echo.
echo            0       0.94      0.94      0.94      5066
echo            1       0.79      0.80      0.80      1418
echo.
echo     accuracy                           0.91      6484
echo    macro avg       0.87      0.87      0.87      6484
echo weighted avg       0.91      0.91      0.91      6484
echo ```
echo.
echo ---
echo.
echo ## 7. Interpretation
echo.
echo The baseline model achieves a ROC-AUC of **0.9438**, indicating strong ranking performance on the held-out test set.
echo.
echo The resulting Gini coefficient of **0.8876** is substantially higher than the project's minimum acceptance criterion of **0.45**.
echo.
echo The model achieves:
echo.
echo - **91.10%% accuracy**
echo - **79.47%% precision**
echo - **79.97%% recall**
echo - **79.72%% F1-score**
echo - **94.38%% ROC-AUC**
echo - **88.76%% Gini**
echo.
echo ---
echo.
echo ## 8. Baseline Conclusion
echo.
echo The baseline XGBoost model successfully meets the project's primary performance requirement.
echo.
echo ```text
echo Required Gini : >= 0.45
echo Baseline Gini : 0.8876
echo Status        : PASS
echo ```
echo.
echo The baseline will be used as the reference benchmark for subsequent model development.
echo.
echo ---
echo.
echo ## 9. Planned Improvements
echo.
echo 1. Hyperparameter tuning
echo 2. Feature engineering
echo 3. Model calibration
echo 4. SHAP-based explainability
echo 5. Individual prediction explanations
echo 6. Risk-band classification
echo 7. Age-group fairness analysis
echo 8. API integration
echo 9. API load testing
echo 10. React frontend integration
echo.
echo ---
echo.
echo ## 10. Reproducibility
echo.
echo ```bash
echo python src/train.py
echo python src/evaluate.py
echo ```
echo.
echo Model artifacts are stored under:
echo.
echo `models/`
echo.
echo Evaluation reports are stored under:
echo.
echo `reports/`
echo.
echo ---
echo.
echo ## 11. Project Acceptance Status
echo.
echo | Acceptance Requirement | Result | Status |
echo |---|---:|---|
echo | XGBoost baseline model | Implemented | ✅ |
echo | ROC-AUC evaluation | 0.9438 | ✅ |
echo | Gini >= 0.45 | 0.8876 | ✅ |
echo | Precision evaluation | 0.7947 | ✅ |
echo | Recall evaluation | 0.7997 | ✅ |
echo | F1 evaluation | 0.7972 | ✅ |
echo | Baseline report | Completed | ✅ |
) > reports\baseline_report.md

echo.
echo ==========================================
echo BASELINE REPORT CREATED SUCCESSFULLY
echo ==========================================
echo.
type reports\baseline_report.md
echo.
echo ==========================================
echo GIT STATUS
echo ==========================================
git status
