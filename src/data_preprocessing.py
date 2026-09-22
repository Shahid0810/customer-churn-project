import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/customer_churn.csv")

print("Dataset loaded successfully!")
print("Original shape:", df.shape)


# ==========================================
# 2. REMOVE CUSTOMER ID
# ==========================================

df = df.drop("customerID", axis=1)

print("\nCustomerID removed.")
print("Shape after removing ID:", df.shape)


# ==========================================
# 3. CONVERT TOTALCHARGES TO NUMERIC
# ==========================================

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)


# ==========================================
# 4. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 5. FILL MISSING TOTALCHARGES
# ==========================================

df["TotalCharges"] = df["TotalCharges"].fillna(0)

print("\nMissing values after cleaning:")
print(df.isnull().sum().sum())


# ==========================================
# 6. CONVERT TARGET VARIABLE
# ==========================================

df["Churn"] = df["Churn"].map({
    "Yes": 1,
    "No": 0
})


# ==========================================
# 7. DISPLAY FINAL DATA
# ==========================================

print("\nFinal dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nChurn distribution:")
print(df["Churn"].value_counts())