# =======================================================
# preprocessing.py -- Data Cleaning and Preparation
# Mohammad Idrees  | GCET Kashmir | midrees-ai
# =======================================================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv(r'c:\users\moham\onedrive\desktop\dsek-eye-health-tracker\eye_data.csv')

# ------ STEP 1: Check missing values ------
print("Missing values:")
print(df.isnull().sum)

# ------ STEP 2: Encode text columns ------
le = LabelEncoder()
df['hospital_encoded'] = le.fit_transform(df['hospital'])
df['vision_encoded'] = le.fit_transform(df['vision_left'])
df['cornea_encoded'] = le.fit_transform(df['cornea_finding'])
df['bcl_status_encoded'] = le.fit_transform(df['bcl_status'])
df['phase_encoded'] = le.fit_transform(df['phase'])

# ------ STEP 3: Encode targets ------
df['iop_risk_encoded'] = le.fit_transform(df['iop_risk'])
df['recovery_encoded'] = le.fit_transform(df['recovery_stage'])

print("\nEncoding done. Unique values:")
print('IOP Risk:', df['iop_risk'].unique())
print('Recovery Stage:', df['recovery_stage'].unique())

# ------ STEP 4: Define features ------
features = ["days_from_surgery", "iop_left", "pain_level", "num_medicines", "has_antifungal", "has_antiviral", "has_iop_drop", "has_steroid", "has_antibiotic", "hospital_encoded", "vision_encoded","cornea_encoded", "bcl_status_encoded", "phase_encoded" ]

X = df[features]
y_iop = df['iop_risk_encoded']
y_recovery = df['recovery_encoded']

print(f"\nFeatures shape: {X.shape}")
print(f"IOP Risk target shape: {y_iop.shape}")
print(f"Recovery target shape: {y_recovery.shape}")

# ------ STEP 5: Train/Test Split ------
X_train, X_test, y_iop_train, y_iop_test = train_test_split(X, y_iop, test_size=0.2, random_state=42)

X_train2, X_test2, y_rec_train, y_rec_test = train_test_split(X, y_recovery, test_size=0.2, random_state=42)

print(f"Training size: {X_train.shape[0]} rows")
print(f"Testing size: {X_test.shape[0]} rows")
print("\nPreprocessing complete!")