import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\LENOVO\Downloads\LoanApprovalProject\train.csv")

print(df.head())
print("\nShape:")
print(df.shape)
print("\nMissing Values:")
print(df.isnull().sum())
print("\nLoan Status Count:")
print(df["Loan_Status"].value_counts())
#data cleaning
# Fill categorical columns with mode
df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])
df["Married"] = df["Married"].fillna(df["Married"].mode()[0])
df["Education"] = df["Education"].fillna(df["Education"].mode()[0])
df["Employment_Status"] = df["Employment_Status"].fillna(
    df["Employment_Status"].mode()[0]
)
# Fill numerical columns with median
df["Applicant_Income"] = df["Applicant_Income"].fillna(
    df["Applicant_Income"].median()
)
df["Loan_Amount"] = df["Loan_Amount"].fillna(
    df["Loan_Amount"].median()
)
print(df.isnull().sum())
df.drop("Loan_ID", axis=1, inplace=True)
print("Duplicates:", df.duplicated().sum())

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in df.select_dtypes(include="object").columns:
    df[col] = le.fit_transform(df[col])

##Feature and Target Split
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
from sklearn.metrics import accuracy_score
## random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, pred_rf)
print("Random Forest Accuracy:", acc_rf)

#XG boost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
xgb.fit(X_train, y_train)
pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, pred_xgb)
print("XGBoost Accuracy:", acc_xgb)


import joblib
joblib.dump(xgb, "loan_model.pkl")
print("Model Saved Successfully")