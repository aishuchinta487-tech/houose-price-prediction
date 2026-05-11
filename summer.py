import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("kc_house_data.csv")
print(df.head())

df.isnull().sum()
df.drop_duplicates(inplace=True)
df.drop(columns=['date'], inplace=True)  # string column remove

X = df[["sqft_living15","bedrooms","bathrooms","floors","sqft_above","sqft_lot15","lat","long"]]  # Feature
y = df["price"]            # Target

df.corr()                  # ఇప్పుడు error రాదు

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
model=RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(X_train,y_train)


print("Score:", model.score(X_test, y_test))