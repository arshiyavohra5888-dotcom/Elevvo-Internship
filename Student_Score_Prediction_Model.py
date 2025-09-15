"""
Student Score Prediction – Elevvo Tech Internship
-------------------------------------------------
Predict exam scores from hours studied using
Linear Regression and Polynomial Regression.
Includes interactive testing and a final
Predicted vs Actual graph.
"""

# ------------------------- 1. Imports -------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------- 2. Load & Prepare Data -------------
df = pd.read_csv("StudentPerformanceFactors.csv", encoding="latin1")

# Standardize column names
df.rename(columns=lambda x: x.strip().replace(" ", "_").lower(), inplace=True)

# Keep only numeric columns we need
df = df[["hours_studied", "exam_score"]].dropna()

print("Dataset shape:", df.shape)
print(df.head(), "\n")

# ------------------------- 3. Visualize Raw Data --------------
plt.scatter(df["hours_studied"], df["exam_score"], color="blue")
plt.title("Study Hours vs Exam Score")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()

# ------------------------- 4. Train/Test Split ----------------
X = df[["hours_studied"]]
y = df["exam_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------- 5. Linear Regression ---------------
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

print("Linear Regression R²:", round(r2_score(y_test, y_pred), 3))
print("Linear Regression RMSE:",
      round(mean_squared_error(y_test, y_pred, squared=False), 3))
# Plot predictions vs actual
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred, color="red", linewidth=2, label="Predicted")
plt.title("Linear Regression: Actual vs Predicted")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()


# ------------------------- 6. Polynomial Regression -----------
poly_model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
poly_model.fit(X_train, y_train)
poly_pred = poly_model.predict(X_test)

print("Polynomial Regression (deg 2) R²:",
      round(r2_score(y_test, poly_pred), 3))
print("Polynomial Regression (deg 2) RMSE:",
      round(mean_squared_error(y_test, poly_pred, squared=False), 3))

# Plot polynomial curve
X_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
plt.scatter(X, y, color="blue", alpha=0.5)
plt.plot(X_range, poly_model.predict(X_range), color="green", linewidth=2)
plt.title("Polynomial Regression Curve")
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.show()


# ------------------------- 7. Interactive Client Testing ------
def predict_linear(hours: float) -> float:
    """Predict score with Linear Regression."""
    sample = pd.DataFrame([[hours]], columns=["hours_studied"])
    return float(lin_reg.predict(sample)[0])

def predict_poly(hours: float) -> float:
    """Predict score with Polynomial Regression."""
    return float(poly_model.predict([[hours]])[0])

try:
    user_hours = float(input("\nEnter study hours to predict the exam score: "))
    pred_lin = predict_linear(user_hours)
    pred_poly = predict_poly(user_hours)

    print(f"\nLinear model prediction for {user_hours} hours: {pred_lin:.2f}")
    print(f"Polynomial model prediction for {user_hours} hours: {pred_poly:.2f}")

    # Append user test point to compare on graph
    plt.scatter(X_test, y_test, color="blue", label="Actual (Test)")
    plt.scatter(X_test, y_pred, color="red", alpha=0.6, label="Linear Pred")
    plt.scatter(user_hours, pred_lin, color="black", s=100,
                edgecolors="yellow", label="Your Linear Input")

    plt.title("Predicted vs Actual (Linear Model)")
    plt.xlabel("Hours Studied")
    plt.ylabel("Exam Score")
    plt.legend()
    plt.show()

except ValueError:
    print("Invalid input. Please enter a numeric value for hours.")
# ------------------------- 8. Final Visualization -------------
plt.scatter(df["hours_studied"], df["exam_score"], color="blue", label="Actual Data")
plt.scatter(X_test, y_pred, color="red", alpha=0.6, label="Linear Pred")
plt.scatter(X_test, poly_pred, color="green", alpha=0.6, label="Poly Pred")
plt.title("Predicted vs Actual Exam Scores")