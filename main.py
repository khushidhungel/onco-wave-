import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from scipy.fft import fft

# Load data
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Add FFT features
fft_values = np.abs(fft(X.values, axis=1))
fft_df = pd.DataFrame(fft_values,
                       columns=[f"fft_{i}" for i in range(X.shape[1])])
X = pd.concat([X, fft_df], axis=1)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("OncoWave Accuracy:", round(accuracy * 100, 2), "%")
print(classification_report(y_test, model.predict(X_test),
      target_names=['Malignant', 'Benign']))

# ─── PREDICTION SYSTEM ────────────────────────────────────
print("\n" + "="*50)
print("       ONCOWAVE — PATIENT ANALYSIS SYSTEM")
print("="*50)

# Use a real sample from dataset (sample number 0)
sample_index = 0
raw_sample = data.data[sample_index]
actual_label = data.target[sample_index]

# Convert to dataframe
sample_df = pd.DataFrame([raw_sample], columns=data.feature_names)

# Add FFT features to sample
sample_fft = np.abs(fft(raw_sample))
sample_fft_df = pd.DataFrame(
    [sample_fft],
    columns=[f"fft_{i}" for i in range(len(raw_sample))]
)
sample_full = pd.concat([sample_df, sample_fft_df], axis=1)

# Predict
prediction = model.predict(sample_full)[0]
probability = model.predict_proba(sample_full)[0]

# Output
print(f"\nSample #{sample_index} Analysis:")
print(f"Actual diagnosis : {'Benign' if actual_label == 1 else 'Malignant'}")
print(f"OncoWave says    : {'Benign' if prediction == 1 else 'Malignant'}")
print(f"Confidence       : {round(max(probability) * 100, 2)}%")

if prediction == actual_label:
    print("Result           : CORRECT ✓")
else:
    print("Result           : INCORRECT ✗")
print("="*50)