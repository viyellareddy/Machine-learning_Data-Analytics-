
# Automatidata Project: The Nuts and Bolts of Machine Learning

### Project Overview
This project involves building a machine learning model to predict customer tipping behavior in taxi rides. The goal is to identify potential generous tippers to help taxi drivers maximize their earnings. The project includes steps for data preparation, feature engineering, model building, and evaluation, using tree-based modeling techniques.

- **Objective**: Predict if a customer will leave a generous tip.
- **Implications**: The model's errors can have consequences:
  - **False Negatives**: Drivers might miss out on potential generous tippers.
  - **False Positives**: Drivers might expect a generous tip but receive none, leading to frustration.
- **Decision**: Instead of predicting non-tippers, the model focuses on identifying generous tippers to avoid potential bias and ensure equitable access to taxi services.

## Feature Engineering
- **Data Preparation**:
  - Loaded datasets: `2017_Yellow_Taxi_Trip_Data.csv` and `nyc_preds_means.csv`.
  - Filtered data to include only credit card payments.
- **Feature Creation**:
  - Created `tip_percent` and `generous` columns to identify generous tippers.
  - Extracted day of the week and time of day from pickup datetime.
  - Binned time of day into `am_rush`, `daytime`, `pm_rush`, and `nighttime`.
  - Extracted month from pickup datetime.
- **Data Cleaning**:
  - Dropped irrelevant columns.
  - Converted categorical variables to binary using one-hot encoding.

## Modeling
- **Data Split**: Split data into training and testing sets (80/20).
- **Models Used**:
  - **Random Forest**: Tuned using GridSearchCV with cross-validation.
  - **XGBoost**: Tuned using GridSearchCV with cross-validation.
- **Evaluation Metrics**: Focused on F1 score to balance precision and recall.
- **Results**:
  - **Random Forest**: Best F1 score on test data: 0.7235.
  - **XGBoost**: Best F1 score on test data: 0.709982.
- **Confusion Matrix**: Evaluated type I and type II errors.

## Conclusion
- **Recommendation**: Use the Random Forest model due to its slightly better performance.
- **Insights**:
  - VendorID, predicted fare, mean duration, and mean distance are key features.
  - Future improvements could include more granular features and historical tipping data.

