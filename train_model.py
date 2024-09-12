import joblib
import pandas as pd
import numpy as np
#from sklearn.datasets import load_boston
from sklearn.datasets import fetch_california_housing # Use fetch_california_housing instead of load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

#boston = load_boston()
housing = fetch_california_housing() # Use fetch_california_housing instead of load_boston
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['PRICE'] = housing.target

X = data.drop('PRICE', axis=1)
y = data['PRICE']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

joblib.dump(model, 'house_price_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
