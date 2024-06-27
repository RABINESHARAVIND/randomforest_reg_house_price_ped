from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#from sklearn.metrics import mean_squared_error
import pandas as pd
import pickle

file_path = "D:\\timpro\\regression-prediction-using-flask\\house_data.csv"
df = pd.read_csv(file_path)

columns = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'price']
df = df[columns]

df = df.dropna()

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

min_max_scaler = MinMaxScaler().fit(X_train)
standard_scaler = StandardScaler().fit(min_max_scaler.transform(X_train))

X_train_normalized = min_max_scaler.transform(X_train)
X_train_standardized = standard_scaler.transform(X_train_normalized)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train_standardized, y_train)

X_test_normalized = min_max_scaler.transform(X_test)
X_test_standardized = standard_scaler.transform(X_test_normalized)
y_pred = rf.predict(X_test_standardized)

#mse = mean_squared_error(y_test, y_pred)
#print(f"Mean Squared Error: {mse}")

with open('model.pkl', 'wb') as f:
    pickle.dump(rf, f)

with open('min_max_scaler.pkl', 'wb') as f:
    pickle.dump(min_max_scaler, f)

with open('standard_scaler.pkl', 'wb') as f:
    pickle.dump(standard_scaler, f)
