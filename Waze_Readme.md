
# Waze User Churn Prediction Project: The Nuts and Bolts of Machine Learning

### Project Overview
This project involves building a machine learning model to predict user churn for Waze. The goal is to help Waze prevent user churn, improve user retention, and grow its business. The project includes ethical considerations, feature engineering, model building, and evaluation using tree-based models: Random Forest and XGBoost.

- **Objective**: Predict if a Waze user will churn or be retained.
- **Implications**: The model's errors can have consequences:
  - **False Negatives**: Users likely to churn may not receive proactive measures to retain them.
  - **False Positives**: Loyal users may receive unnecessary retention measures, potentially leading to a negative experience.
- **Decision**: Proceed with building the model due to minimal significant risks.

## Feature Engineering
- **Data Preparation**:
  - Loaded dataset: `waze_dataset.csv`.
  - Handled missing values by dropping rows with missing data in the `label` column.
- **Feature Creation**:
  - `km_per_driving_day`: Mean kilometers driven per driving day.
  - `percent_sessions_in_last_month`: Percentage of total sessions logged in the last month.
  - `professional_driver`: Binary feature for users with 60+ drives and 15+ driving days in the last month.
  - `total_sessions_per_day`: Mean number of sessions per day since onboarding.
  - `km_per_hour`: Mean kilometers per hour driven in the last month.
  - `km_per_drive`: Mean kilometers per drive in the last month.
  - `percent_of_drives_to_favorite`: Percentage of total sessions navigating to favorite places.
- **Data Cleaning**:
  - Converted infinite values to zero.
  - Encoded categorical variables and created dummy variables.
- **Feature Selection**:
  - Dropped irrelevant columns (`ID`, `device`).

## Modeling
- **Data Split**: Split data into training (60%), validation (20%), and testing (20%) sets.
- **Models Used**:
  - **Random Forest**: Tuned using GridSearchCV. Achieved a recall score of 0.127.
  - **XGBoost**: Tuned using GridSearchCV. Achieved a recall score of 0.173.
- **Evaluation Metrics**: Focused on recall to ensure all potential churns are identified.
- **Results**:
  - The XGBoost model performed better than the Random Forest model in recall.
  - Final XGBoost model: F1 score, precision, and recall were evaluated on both validation and test sets.

## Conclusion
- **Recommendation**: Use the XGBoost model for churn prediction due to its superior performance in recall.
- **Insights**:
  - Engineered features significantly contributed to the model's predictive power.
  - The model successfully identified factors driving user churn, with engagement metrics being highly predictive.

