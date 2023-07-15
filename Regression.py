from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import numpy as npz
import pandas as pd
import seaborn as sn
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

fname = r"C:\Users\HP\PycharmProjects\ObesityEstimation_OPER5151\DATA_BMI.csv"
df = pd.read_csv(fname)

X = df.drop(['BMI'], axis=1)
y = df['BMI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Plotting the predicted values versus the actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual BMI')
plt.ylabel('Predicted BMI')
plt.title('Actual vs Predicted BMI')

# Add a diagonal line for reference
min_val = min(min(y_test), min(y_pred))
max_val = max(max(y_test), max(y_pred))
plt.plot([min_val, max_val], [min_val, max_val], 'r--')

# Show the plot (LOOK
plt.show()
