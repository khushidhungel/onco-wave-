import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy.fft import fft

# ── Train model ──────────────────────────────────────
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

fft_values = np.abs(fft(X.values, axis=1))
fft_df = pd.DataFrame(fft_values,
                       columns=[f"fft_{i}" for i in range(X.shape[1])])
X_full = pd.concat([X, fft_df], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_full, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

# ── UI ───────────────────────────────────────────────
st.set_page_config(page_title="OncoWave", page_icon="🧬")

st.title("🧬 OncoWave")
st.subheader("AI Cancer Detection System")
st.markdown(f"**Model Accuracy: {round(accuracy * 100, 2)}%**")
st.divider()

st.markdown("### Enter Patient Cell Measurements")

# Input sliders for top 5 most important features
feature_names = data.feature_names
col1, col2 = st.columns(2)

inputs = {}
for i, feature in enumerate(feature_names[:15]):
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    if i % 2 == 0:
        inputs[feature] = col1.slider(
            feature, min_val, max_val, mean_val
        )
    else:
        inputs[feature] = col2.slider(
            feature, min_val, max_val, mean_val
        )

# Fill remaining features with mean
full_input = []
for feature in feature_names:
    if feature in inputs:
        full_input.append(inputs[feature])
    else:
        full_input.append(float(X[feature].mean()))

# Predict button
if st.button("🔬 Analyze Patient"):
    raw = np.array(full_input)
    sample_fft = np.abs(fft(raw))
    sample_full = np.concatenate([raw, sample_fft])
    sample_df = pd.DataFrame([sample_full],
                              columns=X_full.columns)

    prediction = model.predict(sample_df)[0]
    probability = model.predict_proba(sample_df)[0]
    confidence = round(max(probability) * 100, 2)

    st.divider()
    if prediction == 1:
        st.success(f"✅ Result: BENIGN — Confidence: {confidence}%")
    else:
        st.error(f"⚠️ Result: MALIGNANT — Confidence: {confidence}%")

    st.markdown(f"**Malignant probability:** {round(probability[0]*100,2)}%")
    st.markdown(f"**Benign probability:** {round(probability[1]*100,2)}%")
