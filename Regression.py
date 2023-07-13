from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from Transformation import transformation

df = transformation()
X = df.drop(['BMI'], axis=1)
y = df['BMI']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)