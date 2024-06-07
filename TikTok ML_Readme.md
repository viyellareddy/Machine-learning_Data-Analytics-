
# TikTok Project: The Nuts and Bolts of Machine Learning

### Project Overview
This project involves building a machine learning model to predict whether a TikTok video contains a claim or an opinion. The goal is to help TikTok reduce the backlog of user reports and prioritize videos for review more efficiently. The project includes ethical considerations, feature engineering, model building, and evaluation.

- **Objective**: Predict if a TikTok video contains a claim or an opinion.
- **Implications**: The model's errors can have consequences:
  - **False Negatives**: Claims misclassified as opinions may go unreviewed, potentially allowing violations of terms of service.
  - **False Positives**: Opinions misclassified as claims may undergo unnecessary review.
- **Decision**: Prioritize recall to minimize false negatives, ensuring that potential violations are reviewed.

## Feature Engineering
- **Data Preparation**:
  - Loaded dataset: `tiktok_dataset.csv`.
  - Handled missing values by dropping rows with missing data.
  - Verified that there were no duplicates.
- **Feature Creation**:
  - Extracted text length from video transcriptions.
  - Tokenized video transcriptions into 2-grams and 3-grams.
  - Encoded categorical variables and created dummy variables.
- **Data Cleaning**:
  - Dropped irrelevant columns and performed encoding to prepare data for modeling.

## Modeling
- **Data Split**: Split data into training (60%), validation (20%), and testing (20%) sets.
- **Models Used**:
  - **Random Forest**: Tuned using GridSearchCV with cross-validation. Achieved a recall score of 0.995.
  - **XGBoost**: Tuned using GridSearchCV with cross-validation. Achieved a recall score of 0.991.
- **Evaluation Metrics**: Focused on recall to ensure all claims are identified.
- **Results**:
  - Both models performed exceptionally well, with the Random Forest model slightly outperforming XGBoost.
  - Final Random Forest model: F1 score, precision, and recall were nearly perfect on both validation and test sets.

## Conclusion
- **Recommendation**: Use the Random Forest model due to its superior performance in recall.
- **Insights**:
  - The most predictive features were related to user engagement metrics (views, likes, shares, downloads).
  - The model successfully differentiates between claims and opinions with high accuracy.

