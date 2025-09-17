# Task 2: Loan Approval Prediction- Elevvo Tech Internship
# ---------------------------------------------------------------------
# Build and evaluate ML models to predict loan approval.
# Includes interactive function for client-side testing.

# ------------------------- 1. Imports -------------------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt
from pandas.plotting import table
from matplotlib import patches


# ------------------------- 2. Load & Clean Data ---------------
df = pd.read_csv(r"C:\Users\arshi\OneDrive\Desktop\Internship_Material\loan_approval_dataset.csv")

# Clean column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print("Cleaned Columns:", df.columns.tolist(), "\n")

# Target mapping
df["loan_status"] = df["loan_status"].str.strip().map({"Approved": 1, "Rejected": 0})
df.drop(columns=["loan_id"], inplace=True)

X = df.drop(columns=["loan_status"])
y = df["loan_status"]

# Identify numeric/categorical columns
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols, "\n")

# ------------------------- 3. Preprocessing -------------------
numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols)
])

# ------------------------- 4. Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Class balance before SMOTE:\n", y_train.value_counts(normalize=True), "\n")

# ------------------------- 5. Balance Data (SMOTE) ------------
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_prep, y_train)
print("Class balance after SMOTE:\n", pd.Series(y_train_bal).value_counts(normalize=True), "\n")

# ------------------------- 6. Train Models --------------------
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_bal, y_train_bal)

dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train_bal, y_train_bal)

# ------------------------- 7. Evaluation ----------------------
def evaluate(model, name):
    y_pred = model.predict(X_test_prep)
    print(f"--- {name} ---")
    print("Precision:", round(precision_score(y_test, y_pred), 3))
    print("Recall:   ", round(recall_score(y_test, y_pred), 3))
    print("F1-score: ", round(f1_score(y_test, y_pred), 3))
    print(classification_report(y_test, y_pred), "\n")

evaluate(log_reg, "Logistic Regression")
evaluate(dec_tree, "Decision Tree")

# ------------------------- 8. Save Models ---------------------
joblib.dump({"preprocessor": preprocessor, "model": log_reg}, "loan_model_lr.joblib")
joblib.dump({"preprocessor": preprocessor, "model": dec_tree}, "loan_model_dt.joblib")
print("Models saved: loan_model_lr.joblib, loan_model_dt.joblib\n")

# ------------------------- 9. Interactive Client Testing ------
def predict_single(sample: dict, model_path="loan_model_dt.joblib") -> dict:
    """
    Predict approval for a single loan application.
    sample: dict with all feature_name: value pairs.
    """
    saved = joblib.load(model_path)
    preproc = saved["preprocessor"]
    model = saved["model"]
    sample_df = pd.DataFrame([sample])
    X_sample = preproc.transform(sample_df)
    pred = int(model.predict(X_sample)[0])
    proba = float(model.predict_proba(X_sample)[0][1])
    return 
import pandas as pd

# Load pipeline dict (preprocessor + chosen model)
saved = joblib.load("loan_model_dt.joblib")   # or loan_model_lr.joblib
preproc = saved["preprocessor"]
model   = saved["model"]

print("\n--- Loan Application Entry ---")
sample = {}

# Collect user input
sample["no_of_dependents"] = int(input("Number of dependents: "))


# ------------------------- 10. Interactive Console Prediction -------------------------
import joblib
sample["education"]               = input("Education (Graduate / Not Graduate): ").strip()
sample["self_employed"]           = input("Self employed? (Yes / No): ").strip()
sample["income_annum"]            = float(input("Annual income (₹): "))
sample["loan_amount"]             = float(input("Loan amount (₹): "))
sample["loan_term"]               = int(input("Loan term (in months): "))
sample["cibil_score"]             = float(input("CIBIL score (300–900): "))
sample["residential_assets_value"]= float(input("Residential assets value (₹): "))
sample["commercial_assets_value"] = float(input("Commercial assets value (₹): "))
sample["luxury_assets_value"]     = float(input("Luxury assets value (₹): "))
sample["bank_asset_value"]        = float(input("Bank asset value (₹): "))

# Convert to DataFrame
sample_df = pd.DataFrame([sample])

# Predict
X_sample = preproc.transform(sample_df)
pred     = int(model.predict(X_sample)[0])
prob     = model.predict_proba(X_sample)[0][1]

# Prepare a result table
result_table = pd.DataFrame({
    "Feature": list(sample.keys()) + ["Prediction", "Approval Probability"],
    "Value":   list(sample.values()) + [
        "Approved ✓" if pred == 1 else "Not Approved ✗",
        f"{prob:.2%}"
    ]
})


print("\n--- Prediction Result ---")
# ------------------------- 11. Visual Table + Popup -------------------------
# ---------- Build a pretty “dashboard” table ----------
fig, ax = plt.subplots(figsize=(9, 6))
ax.axis("off")

tbl = table(ax, result_table, loc="center", cellLoc="center", colWidths=[0.4, 0.4])
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1.3, 1.3)

# Highlight key rows
for (r, c), cell in tbl.get_celld().items():
    if r > 0:  # skip header row
        feature = result_table.iloc[r - 1]["Feature"]
        if feature == "Prediction" and c == 1:
            cell.set_facecolor("#ffe082")  # yellow highlight
            cell.set_text_props(weight="bold", color="black")
        elif feature == "Approval Probability" and c == 1:
            cell.set_facecolor("#c5e1a5" if pred == 1 else "#ef9a9a")  # green/red
            cell.set_text_props(weight="bold", color="black")

ax.set_title("Loan Application Summary", fontsize=14, weight="bold", pad=20)
plt.tight_layout()
plt.show()