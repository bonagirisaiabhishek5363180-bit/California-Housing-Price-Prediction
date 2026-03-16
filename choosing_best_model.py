import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error,classification_report
df=pd.read_csv("housing.csv")
df.head()
df.info()
df.isnull().sum()
df.describe()
df['median_income_cat']=pd.cut(df["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5]).copy()
split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index,test_index in split.split(df,df["median_income_cat"]):
    train_set=df.loc[train_index]
    test_set=df.loc[test_index]
test_set.info()
train_set.info()
housing=train_set.copy()
housing=housing.drop("median_income_cat",axis=1)
print(housing)
housing_labels=housing["median_house_value"].copy()
housing=housing.drop("median_house_value",axis=1)
housing=housing.reset_index(drop=True)
housing.isnull().sum()
housing_num=list(housing.select_dtypes(include=np.number).columns)
housing_cat=["ocean_proximity"]
num_pipeline=Pipeline([
    ("impute",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
cat_pipeline=Pipeline([
    ("OneHot",OneHotEncoder(handle_unknown="ignore"))
])
full_pipeline=ColumnTransformer([
    ("num",num_pipeline,housing_num),
    ("cat",cat_pipeline,housing_cat)
])
housing_prepared = full_pipeline.fit_transform(housing)
linear_model=LinearRegression()
linear_model.fit(housing_prepared,housing_labels)
linear_pred=linear_model.predict(housing_prepared)
linear_rmse=root_mean_squared_error(housing_labels,linear_pred)
print("linear forest rmses")
linear_rmses= -cross_val_score(linear_model,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print(pd.Series(linear_rmses).describe())
desc_model=DecisionTreeRegressor(random_state=42)
desc_model.fit(housing_prepared,housing_labels)
desc_pred=desc_model.predict(housing_prepared)
desc_rmses= -cross_val_score(desc_model,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print("desc forest rmses")
print(pd.Series(desc_rmses).describe())
desc_rmse=root_mean_squared_error(housing_labels,desc_pred)
print(f"descision model rmse:{desc_rmse}")
random_model=RandomForestRegressor(random_state=42)
random_model.fit(housing_prepared,housing_labels)
random_pred=random_model.predict(housing_prepared)
random_rmse=root_mean_squared_error(housing_labels,random_pred)
random_rmses= -cross_val_score(random_model,housing_prepared,housing_labels,scoring="neg_root_mean_squared_error",cv=10)
print("random  forest rmses")
print(pd.Series(random_rmses).describe())
print(f"random model rmse:{random_rmse}")
print(f"The classification report for the random forest model is:\n{classification_report(housing_labels,random_pred)}")
