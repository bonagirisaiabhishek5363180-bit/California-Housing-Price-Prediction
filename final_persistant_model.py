import pandas as pd
import numpy as np  
import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit,cross_val_score 
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

model_file="model.pkl"
pipeline_file="pipeline.pkl"

def build_ppeline(num_attributes,cat_attributes):
    num_pipeline=Pipeline([
        ("impute",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("OneHot",OneHotEncoder(handle_unknown="ignore"))
    ])
    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attributes),
        ("cat",cat_pipeline,cat_attributes)
    ])
    return full_pipeline

if not os.path.exists(model_file):
    #training the model
    df=pd.read_csv("housing.csv")
    df['median_income_cat']=pd.cut(df["median_income"],bins=[0,1.5,3,4.5,6,np.inf],labels=[1,2,3,4,5]).copy()
    split=StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index,test_index in split.split(df,df["median_income_cat"]):
        train_set=df.loc[train_index].drop("median_income_cat",axis=1)
        test_set=df.loc[test_index]
    test_set.to_csv("test_set.csv",index=False)
    housing=train_set.copy()
    housing_features=housing.drop('median_house_value',axis=1).copy()
    housing_labels=housing['median_house_value'].copy()
    num_attributed=list(housing_features.select_dtypes(include=np.number).columns)
    cat_attributes=["ocean_proximity"]
    full_pipeline=build_ppeline(num_attributed,cat_attributes)
    housing_prepared = full_pipeline.fit_transform(housing_features)
    model=RandomForestRegressor(random_state=42)
    model.fit(housing_prepared,housing_labels)
    joblib.dump(model,model_file)
    joblib.dump(full_pipeline,pipeline_file)
else:
    # inferance phase
    model=joblib.load(model_file)
    full_pipeline=joblib.load(pipeline_file)
    input_csv=input("Enter the input csv file:")
    input_df=pd.read_csv(input_csv)
    input_prepared=full_pipeline.transform(input_df)
    predictions=model.predict(input_prepared)
    input_df["predictions"]=predictions
    output_file=input("Enter the output csv file to save predictions:")
    input_df.to_csv(output_file,index=False)
    print(f"predictions saved to {output_file}")    