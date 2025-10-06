import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ydata_profiling import ProfileReport
df=pd.read_csv("housing.csv")
df.head()
df.info()
df.describe()
df.isnull().sum()   
df.hist(bins=50,figsize=(20,15))
plt.show()
# gives a detailed eda report
report=ProfileReport(df,title="Housing Data Report",explorative=True)
report.to_notebook_iframe()
# save the report as html file
report.to_file("housing_data_report.html")