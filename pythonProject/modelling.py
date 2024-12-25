import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

import matplotlib.pyplot as plt
import seaborn as sns

# load data
file_path = "./data/final.csv"
df_final = pd.read_csv(file_path)
print(f"\nshape of df: {df_final.shape}")

# separate X and y
X, y = df_final.iloc[:, :-1], df_final.iloc[:, -1]

# size
print(f"\nshape of X: {X.shape}")
print(f"shape of y: {y.shape}")

## split into train and test dataset
TEST_SIZE = 0.3
RANDOM_STATE = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# scaling dataset
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# save scaling
pickle.dump(scaler, open("models/scaling.pkl", "wb"))

# modelling
regression = LinearRegression()
regression.fit(X_train, y_train)

# model parameters
print("\nCoefficients: ", regression.coef_)
print("\nIntercept: ", regression.intercept_)

# get model params
print("\nModel params: ", regression.get_params())

# predict
reg_pred = regression.predict(X_test)

# verify
plt.scatter(y_test, reg_pred)
plt.savefig("plots/test_vs_pred.png")

# residual
residuals = y_test - reg_pred
sns.displot(residuals, kind="kde")
plt.savefig("plots/residuals.png")

# scatter residuals
plt.scatter(reg_pred, residuals)
plt.savefig("plots/residuals_scatter.png")

# accuracy
print("\nMean absolute error: ", mean_absolute_error(y_test, reg_pred))
print("Mean squared error: ", mean_squared_error(y_test, reg_pred))
print("Root mean squared (RMS) error: ", np.sqrt(mean_squared_error(y_test, reg_pred)))

# r2 score
score = r2_score(y_test, reg_pred)
adjusted_r2_score = 1 - (1 - score) * (len(y_test) - 1) / (
    len(y_test) - X_test.shape[1] - 1
)
print(f"\nr2 Score: {score}")
print(f"Adjusted r2 Score: {adjusted_r2_score}")

# pickle models
pickle.dump(regression, open("models/regmodel.pkl", "wb"))
