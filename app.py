import streamlit as st
import numpy as np
import pandas as pd
import pickle

# ===============================
# Page config
# ===============================
st.set_page_config(
    page_title="Predictive Maintenance System",
    page_icon="🛠️",
    layout="wide"
)

# ===============================
# Load model & preprocessors
# ===============================
model = pickle.load(open("bagging_tomek_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))
robust_scaler = pickle.load(open("robust_scaler.pkl", "rb"))
minmax_scaler = pickle.load(open("minmax_scaler.pkl", "rb"))

# ===============================
# Sidebar navigation
# ===============================
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Project Overview", "Failure Prediction"]
)

# ===============================
# PAGE 1: Overview
# ===============================
if page == "Project Overview":

    st.title("🛠️ Predictive Maintenance Using Machine Learning")

    st.markdown("""
    ### 📌 Objective
    This system predicts **machine failure events** using historical sensor data.
    The model is trained on the **AI4I 2020 Predictive Maintenance Dataset**.

    ### 🧠 Key Features
    - Handles **severely imbalanced data**
    - Uses **Bagging + Tomek Links**
    - Optimized using **Macro F1 Score**
    - Designed for **early failure detection**

    ### 🏆 Best Model
    **Bagging Classifier with Tomek Links**
    - Macro F1 Score: **0.88**
    - Minority Recall: **0.71**
    - Suitable for high-risk industrial decisions

    ### ⚠️ Why Macro F1?
    Accuracy is misleading for imbalanced datasets.
    Macro F1 ensures **both failure and non-failure cases** are treated fairly.
    """)

# ===============================
# PAGE 2: Prediction UI
# ===============================
elif page == "Failure Prediction":

    st.title("🔧 Machine Failure Prediction")

    st.markdown("Enter current machine sensor readings:")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            ["Low (L)", "Medium (M)", "High (H)"]
        )

        air_temp = st.slider(
            "Air Temperature (K)",
            295.0, 305.0, 300.0
        )

        process_temp = st.slider(
            "Process Temperature (K)",
            305.0, 315.0, 310.0
        )

    with col2:
        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=1000,
            max_value=3000,
            value=1500
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=5.0,
            max_value=80.0,
            value=40.0
        )

        tool_wear = st.slider(
            "Tool Wear (minutes)",
            0, 300, 100
        )

    # ===============================
    # Prediction button
    # ===============================
    if st.button("🔮 Predict Failure"):

        # Encode machine type
        type_map = {"Low (L)": "L", "Medium (M)": "M", "High (H)": "H"}
        type_encoded = encoder.transform([[type_map[machine_type]]])[0][0]

        # Scale features
        rot_scaled, torque_scaled = robust_scaler.transform(
            [[rotational_speed, torque]]
        )[0]

        air_scaled, proc_scaled, wear_scaled = minmax_scaler.transform(
            [[air_temp, process_temp, tool_wear]]
        )[0]

        # Final feature vector
        X_input = np.array([[ 
            type_encoded,
            rot_scaled,
            torque_scaled,
            air_scaled,
            proc_scaled,
            wear_scaled
        ]])

        prediction = model.predict(X_input)[0]

        st.markdown("---")

        if prediction == 1:
            st.error("🚨 **HIGH RISK: Machine Failure Likely**")
            st.markdown("""
            **Recommended Action**
            - Schedule immediate inspection
            - Reduce machine load
            - Prepare maintenance resources
            """)
        else:
            st.success("✅ **LOW RISK: Machine Operating Normally**")
            st.markdown("""
            **Recommended Action**
            - Continue normal operation
            - Monitor periodically
            """)

