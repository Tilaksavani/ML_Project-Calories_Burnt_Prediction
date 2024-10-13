# ML_Project-Calories_Burnt_Prediction 🔥🏋️‍♂️

This project explores the task of predicting calories burnt during physical activities using an **XGBRegressor** model. By analyzing various attributes related to physical exercise, such as duration, intensity, and personal characteristics, the model aims to estimate the number of calories burnt during an activity.

## Data
This directory contains two datasets (`calories.csv` and `exercise.csv`) used for the project. The datasets include the following features:

### exercise.csv:
- **User_ID**: Unique ID for each user.
- **Gender**: Gender of the user (Male/Female).
- **Age**: Age of the user.
- **Height**: Height of the user (in cm).
- **Weight**: Weight of the user (in kg).
- **Duration**: Duration of the exercise (in minutes).
- **Heart_Rate**: Average heart rate during the exercise.
- **Body_Temp**: Average body temperature during the exercise (in °C).

### calories.csv:
- **User_ID**: Unique ID for each user.
- **Calories_Burnt**: Total calories burnt during the exercise (target variable).

> **Note:** These datasets are merged based on the **User_ID** to analyze and predict the calories burnt during physical activities.

## Notebooks
This directory contains the Jupyter Notebook (`Calories_Burnt_Prediction.ipynb`) that guides you through the entire process of data exploration, preprocessing, model training, evaluation, and visualization.

## Running the Project
The Jupyter Notebook (`Calories_Burnt_Prediction.ipynb`) walks through the following steps:

### Data Loading and Exploration:
- Load both datasets and merge them on **User_ID**.
- Explore basic statistics and relationships between features and the target variable (`Calories_Burnt`).

### Data Preprocessing:
- Handle missing values (if any).
- Scale numerical features like **Age**, **Height**, **Weight**, **Heart_Rate**, and **Body_Temp**.
- Encode categorical variables (e.g., **Gender**).

### Train-Test Split:
- The data is split into training and testing sets using `train_test_split` from the `sklearn` library, with a typical 80-20 or 70-30 ratio for training and testing, respectively.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

### Feature Engineering (Optional):
- Create additional features (e.g., interaction terms between **Heart_Rate** and **Duration**).
- Analyze correlations between features and the target variable.

### Model Training:
- Trains the model using **XGBRegressor**, tuning hyperparameters for improved performance.

```python
from xgboost import XGBRegressor
model = XGBRegressor(n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)
```

### Model Evaluation:
- Evaluates model performance using metrics like **mean absolute error (MAE)**, **R-squared**, and **root mean squared error (RMSE)**.

### Visualization of Results:
- Visualize the relationship between exercise features and calories burnt using various plots.
- Plot feature importance to explore the impact of different features on predictions.

### Customization
- Modify the Jupyter Notebook to:
  - Experiment with different preprocessing techniques and feature engineering methods.
  - Try other regression algorithms for comparison (e.g., **Random Forest Regressor**, **Gradient Boosting**).
  - Explore advanced techniques like deep learning models for regression tasks.

### Resources
- XGBoost Documentation: [https://xgboost.readthedocs.io/en/latest/](https://xgboost.readthedocs.io/en/latest/)
- Kaggle Calories Burnt Dataset: [https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos)

### Further Contributions
- Extend this project by:
  - Incorporating additional data (e.g., types of exercises or workout regimes) for improved prediction.
  - Implementing a real-time calories prediction system using a trained model and an API.
  - Exploring explainability techniques to understand the reasoning behind the model's predictions.

By leveraging machine learning models, specifically **XGBRegressor**, and exercise data, we aim to develop a reliable method for predicting calories burnt. This project lays the foundation for further exploration into health and fitness applications using machine learning.
