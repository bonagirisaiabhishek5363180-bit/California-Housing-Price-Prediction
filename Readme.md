# 🏠 California Housing Price Prediction

An **end-to-end Machine Learning project** that predicts **California
housing prices** using multiple regression models. The project
demonstrates a **complete ML pipeline including Exploratory Data
Analysis (EDA), feature preprocessing, model training, evaluation, and
prediction generation**.

------------------------------------------------------------------------

# 📌 Project Overview

Housing prices depend on several factors such as **income levels,
population density, house age, and geographic location**.

This project builds a **machine learning pipeline** to analyze these
factors and predict **median house values in California districts**.

The workflow follows a **real-world Data Science pipeline used in
industry**:

1.  Data Collection
2.  Data Cleaning
3.  Exploratory Data Analysis (EDA)
4.  Feature Engineering
5.  Data Preprocessing
6.  Model Training
7.  Model Evaluation
8.  Model Selection
9.  Model Persistence
10. Prediction on New Data

------------------------------------------------------------------------

# 🏗️ Machine Learning Pipeline Architecture

Data → Preprocessing → Feature Engineering → Model Training → Evaluation
→ Model Selection → Saved Model → Prediction

------------------------------------------------------------------------

# 📊 Dataset

Dataset: **California Housing Dataset**

Features used:

  Feature              Description
  -------------------- ----------------------------
  longitude            Geographic longitude
  latitude             Geographic latitude
  housing_median_age   Median house age
  total_rooms          Total rooms
  total_bedrooms       Total bedrooms
  population           Population of district
  households           Number of households
  median_income        Median income of residents
  ocean_proximity      Distance from ocean

Target Variable:

`median_house_value`

------------------------------------------------------------------------

# 📈 Exploratory Data Analysis (EDA)

EDA was performed to understand patterns and relationships within the
dataset.

Key insights:

-   **Median income strongly correlates with house prices**
-   Houses closer to the **ocean tend to have higher values**
-   Some features required **scaling and encoding**
-   Missing values were handled during preprocessing

------------------------------------------------------------------------

# ⚙️ Data Preprocessing

A **Scikit-learn pipeline** was created to automate preprocessing steps.

Preprocessing steps include:

-   Handling missing values
-   Standard scaling for numerical features
-   One-hot encoding for categorical features
-   Feature transformation

Libraries used:

pandas\
numpy\
scikit-learn\
matplotlib\
seaborn

------------------------------------------------------------------------

# 🤖 Machine Learning Models

The following models were trained and evaluated:

  Model                     Description
  ------------------------- ---------------------------
  Linear Regression         Baseline regression model
  Decision Tree Regressor   Non-linear model
  Random Forest Regressor   Ensemble model

------------------------------------------------------------------------

# 📊 Model Performance

Model performance was evaluated using **10-fold cross validation**.

### Random Forest Cross Validation Results

  Metric               Value
  -------------------- ------------
  Mean RMSE            **49,432**
  Standard Deviation   2,239
  Minimum RMSE         45,940
  Maximum RMSE         53,301

Summary:

count 10\
mean 49432.12\
std 2239.79\
min 45940.42\
25% 47726.32\
50% 49230.48\
75% 50904.66\
max 53301.08

------------------------------------------------------------------------

# 🧪 Final Test Performance

After selecting the best model, evaluation was performed on the **test
dataset**.

Final Test RMSE = **18,342**

This means the model predicts housing prices with an **average error of
about \$18K**.

------------------------------------------------------------------------

# 🏆 Best Model

Model: Random Forest Regressor\
Evaluation Metric: RMSE\
Test RMSE: **18,342**

------------------------------------------------------------------------

# 📂 Project Structure

California-Housing-Prediction

data/ - housing.csv

notebooks/ - eda_and_model_training.ipynb

models/ - housing_model.pkl

src/ - train.py - preprocessing.py - predict.py

predictions_of_test.csv\
requirements.txt\
README.md

------------------------------------------------------------------------

# 📊 Prediction Output

Predictions are saved into:

`predictions_of_test.csv`

Example output:

Actual Price \| Predicted Price\
500001 \| 483020\
162500 \| 221708\
204600 \| 205706\
159700 \| 170623\
184000 \| 212497

------------------------------------------------------------------------

# 🖥️ Running the Project

Clone the repository

git clone
https://github.com/yourusername/california-housing-prediction.git

Install dependencies

pip install -r requirements.txt

Train the model

python train.py

Run predictions

python final_persistant_model.py

------------------------------------------------------------------------

# 🧠 Skills Demonstrated

-   Exploratory Data Analysis
-   Feature Engineering
-   Machine Learning
-   Model Evaluation
-   Data Preprocessing
-   Scikit-learn Pipelines
-   Model Persistence
-   Prediction Systems

------------------------------------------------------------------------

⭐ If you found this project useful, consider giving it a **star**.
