import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. LOAD DATA
df = pd.read_csv("bitcoin.csv")

# fix tên cột
df.columns = df.columns.str.strip()
print("Columns:", df.columns)

# lấy cột thời gian
date_col = df.columns[0]
df[date_col] = pd.to_datetime(df[date_col])
df = df.sort_values(date_col)

print(df.head())

# 2. CLEAN DATA
df = df.dropna()

# 3. FEATURE + TARGET
X = df[['Open', 'High', 'Low', 'Volume', 
        'MA7', 'MA21', 'Lag_1', 'Lag_2', 'Lag_3',
        'Volatility_7d', 'RSI']]

y = df['Target_3d']

# 4. SPLIT DATA
split = int(len(X) * 0.8)

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

# lấy ngày test
dates = df[date_col]
dates_test = dates[split:]

# 5. TRAIN MODEL
model = LinearRegression()
model.fit(X_train, y_train)

print("\nModel coefficients:")
print(model.coef_)

# 6. PREDICT
y_pred = model.predict(X_test)

# 7. EVALUATE

mse = mean_squared_error(y_test, y_pred)
print("\nMSE:", mse)

# 8. VISUALIZE
plt.figure(figsize=(10,5))

plt.plot(dates_test, y_test.values, label="Actual")
plt.plot(dates_test, y_pred, label="Predicted")

plt.legend()
plt.title("Bitcoin Prediction (3-day ahead)")
plt.xlabel("Date")
plt.ylabel("Price")
plt.xticks(rotation=45)

plt.show()
# 9. PRINT SAMPLE

print("\nFirst 10 predictions vs actual:")
for i in range(10):
    print("Date:", dates_test.iloc[i],
          "Actual:", y_test.values[i],
          "Pred:", y_pred[i])