import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
# Assuming you have a CSV file with columns 'Date' and 'Price'
data = pd.read_csv('gold_price_data.csv')

# Assuming 'Date' column is in datetime format
# Convert it to ordinal (numeric) for regression
data['Date'] = pd.to_datetime(data['Date']).apply(lambda x: x.toordinal())

# Split dataset into features (X) and target variable (y)
X = data[['Date']]  # Assuming only date is used as feature
y = data['Price']

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model initialization and training
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)
