import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, auc
from sklearn.preprocessing import RobustScaler, MinMaxScaler, OrdinalEncoder
import sys
import traceback

# COMPATIBILITY FIX 
def fix_sklearn_model(model):
    """Fix sklearn models for different versions"""
    try:
        if hasattr(model, 'estimators_'):
            for estimator in model.estimators_:
                if hasattr(estimator, 'monotonic_cst'):
                    try:
                        estimator.monotonic_cst = None
                    except:
                        pass
        return model
    except Exception as e:
        return model

def load_model_safe(filepath):
    """Safely load model with compatibility fixes"""
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        model = fix_sklearn_model(model)
        return model
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

st.set_page_config(
    page_title="Predictive Maintenance Dashboard",
    page_icon="⚙️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .cost-savings {
        font-size: 2rem;
        font-weight: bold;
        color: #2ecc71;
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">⚙️ Predictive Maintenance Dashboard</div>', unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        bagging_model = load_model_safe("models/Bagging_binary.pkl")
        rf_model = load_model_safe("models/randomforest_model.pkl")
        balanced_model = load_model_safe("models/BalancedBagging_binary.pkl")
        
        return bagging_model, rf_model, balanced_model
    except Exception as e:
        print(f"Load models error: {e}")
        return None, None, None

bagging_model, rf_model, balanced_bagging_model = load_models()

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio("Select Page", 
                        ["Home", "Model Performance", "Make Predictions", "Cost-Benefit Analysis"])

# Helper function for data preprocessing
def preprocess_input(type_val, air_temp, process_temp, rot_speed, torque_val, tool_wear_val):
    """Preprocess input data exactly as done in training"""
    
    # Encode Type (L=0, M=1, H=2)
    type_map = {'L': 0.0, 'M': 1.0, 'H': 2.0}
    type_encoded = type_map[type_val]
    
    # Scale Rotational speed and Torque with RobustScaler
    # Using approximate scaling parameters from the training data
    # For Rotational speed: median=1503, IQR=189
    # For Torque: median=40.1, IQR=13.6
    rot_speed_scaled = (rot_speed - 1503) / 189
    torque_scaled = (torque_val - 40.1) / 13.6
    
    # Scale other features with MinMaxScaler
    # Air temperature: min=295.3, max=304.5, range=9.2
    # Process temperature: min=305.7, max=313.8, range=8.1
    # Tool wear: min=0, max=253
    air_temp_scaled = (air_temp - 295.3) / 9.2
    process_temp_scaled = (process_temp - 305.7) / 8.1
    tool_wear_scaled = tool_wear_val / 253.0
    
    # dataframe with exact column order from training
    input_df = pd.DataFrame({
        'Type': [type_encoded],
        'Rotational speed': [rot_speed_scaled],
        'Torque': [torque_scaled],
        'Air temperature': [air_temp_scaled],
        'Process temperature': [process_temp_scaled],
        'Tool wear': [tool_wear_scaled]
    })
    
    return input_df

# Cost cal function
def calculate_cost_impact(tp, fp, fn, cost_inspection=5000, cost_breakdown=500000):
    model_cost = ((tp + fp) * cost_inspection) + (fn * cost_breakdown)
    total_failures = tp + fn
    baseline_cost = total_failures * cost_breakdown
    savings = baseline_cost - model_cost
    return baseline_cost, model_cost, savings

# home page
if page == "Home":
    st.write("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📋 Project Overview")
        st.write("""
        This dashboard presents a **Machine Learning based Predictive Maintenance System** 
        designed to identify potential equipment failures before they occur.
        
        ### Key Features:
        - **Real-time Failure Prediction**: Predict machine failures based on operational parameters
        - **Model Comparison**: Evaluate three different ML models
        - **Cost-Benefit Analysis**: Estimate financial savings from predictive maintenance
        - **Interactive Predictions**: Test models with custom input data
        
        ### Dataset Information:
        - **Source**: AI4I 2020 Predictive Maintenance Dataset (UCI ML Repository)
        - **Size**: 10,000 instances
        - **Type**: Multivariate, time series inspired
        - **Target**: Binary classification (failure vs. normal operation)
        """)
    
    with col2:
        st.header("🎯 Models")
        st.info("""
        **1. Bagging with Tomek Links**
        - Best precision-recall balance
        - PR AUC: 0.82
        
        **2. Random Forest with RandomOverSampler**
        - Strong overall performance
        - PR AUC: 0.78
        
        **3. Balanced Bagging**
        - Highest failure detection
        - PR AUC: 0.61
        """)
    
    st.write("---")
    st.header("📊 Dataset Features")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **Operational Parameters:**
        - Air Temperature (K)
        - Process Temperature (K)
        - Rotational Speed (RPM)
        - Torque (Nm)
        - Tool Wear (minutes)
        - Product Type (L/M/H)
        """)
    
    with feature_col2:
        st.markdown("""
        **Target Variable:**
        - Machine Failure (0 = No failure, 1 = Failure)
        
        **Class Distribution:**
        - No Failure: 96.61%
        - Failure: 3.39%
        
        *(Highly imbalanced dataset)*
        """)

# model  performence page
elif page == "Model Performance":
    st.header("📈 Model Performance Comparison")
    
    # Performance metrics data
    models_data = {
        'Model': ['Bagging + Tomek Links', 'Random Forest + RandomOverSampler', 'Balanced Bagging'],
        'Precision': [0.85, 0.85, 0.28],
        'Recall': [0.71, 0.65, 0.88],
        'F1-Score': [0.77, 0.73, 0.42],
        'PR AUC': [0.82, 0.78, 0.61],
        'Accuracy': [0.99, 0.98, 0.92]
    }
    
    df_metrics = pd.DataFrame(models_data)
    
    # Display metrics table
    st.subheader("Performance Metrics Summary")
    st.dataframe(df_metrics.style.highlight_max(axis=0, subset=['Precision', 'Recall', 'F1-Score', 'PR AUC', 'Accuracy']), 
                 use_container_width=True)
    
    st.write("---")
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar chart for comparison
        fig = go.Figure()
        metrics = ['Precision', 'Recall', 'F1-Score', 'PR AUC']
        
        for metric in metrics:
            fig.add_trace(go.Bar(
                name=metric,
                x=df_metrics['Model'],
                y=df_metrics[metric],
                text=df_metrics[metric],
                textposition='auto',
            ))
        
        fig.update_layout(
            title='Model Performance Metrics',
            xaxis_title='Model',
            yaxis_title='Score',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Radar chart
        categories = ['Precision', 'Recall', 'F1-Score', 'PR AUC', 'Accuracy']
        
        fig = go.Figure()
        
        for idx, model in enumerate(df_metrics['Model']):
            fig.add_trace(go.Scatterpolar(
                r=[df_metrics.iloc[idx]['Precision'],
                   df_metrics.iloc[idx]['Recall'],
                   df_metrics.iloc[idx]['F1-Score'],
                   df_metrics.iloc[idx]['PR AUC'],
                   df_metrics.iloc[idx]['Accuracy']],
                theta=categories,
                fill='toself',
                name=model
            ))
        
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title='Model Performance Radar Chart',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    
    # Confusion matrices
    st.subheader("Confusion Matrices")
    
    # Sample confusion matrix values 
    cm_data = {
        'Bagging + Tomek Links': [[2406, 9], [25, 60]],
        'Random Forest + RandomOverSampler': [[2407, 8], [30, 55]],
        'Balanced Bagging': [[2223, 192], [10, 75]]
    }
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (model_name, cm) in enumerate(cm_data.items()):
        with [col1, col2, col3][idx]:
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted No Failure', 'Predicted Failure'],
                y=['Actual No Failure', 'Actual Failure'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            fig.update_layout(
                title=model_name,
                height=300,
                xaxis_title='Predicted',
                yaxis_title='Actual'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    
    # Key insights
    st.subheader("🔍 Key Insights")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("""
        **Best Overall: Bagging + Tomek Links**
        - Highest PR AUC (0.82)
        - Best precision-recall balance
        - Recommended for production
        """)
    
    with col2:
        st.info("""
        **Runner-up: Random Forest**
        - Good overall performance
        - Slightly lower recall
        - Reliable alternative
        """)
    
    with col3:
        st.warning("""
        **High Recall: Balanced Bagging**
        - Catches most failures (88%)
        - Many false alarms (low precision)
        - Use when missing failures is costly
        """)

# prediction page
elif page == "Make Predictions":
    st.header("🔮 Make Predictions")
    
    if bagging_model is None:
        st.error("Models not loaded. Please check model files.")
    else:
        # Instructions
        st.info("📊 **Instructions:** Adjust the sliders below to set machine operational parameters, select a prediction model, and click the Predict button to analyze failure risk.")
        
        st.write("---")
        
        # container for inputs
        with st.container():
            st.subheader("⚙️ Machine Parameters")
            
            # Input sliders in two columns
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("##### 🏭 Configuration")
                product_type = st.selectbox(
                    "Product Type", 
                    ["L", "M", "H"],
                    help="L=Low, M=Medium, H=High quality variant",
                    index=0
                )
                
                st.markdown("##### 🌡️ Temperature")
                air_temp = st.slider(
                    "Air Temperature (K)", 
                    min_value=295.0, 
                    max_value=305.0, 
                    value=300.0, 
                    step=0.1,
                    help="Operating air temperature in Kelvin"
                )
                process_temp = st.slider(
                    "Process Temperature (K)", 
                    min_value=305.0, 
                    max_value=314.0, 
                    value=310.0, 
                    step=0.1,
                    help="Process temperature in Kelvin"
                )
            
            with col2:
                st.markdown("##### ⚡ Mechanical")
                rotational_speed = st.slider(
                    "Rotational Speed (RPM)", 
                    min_value=1100, 
                    max_value=3000, 
                    value=1500, 
                    step=10,
                    help="Machine rotational speed in RPM"
                )
                torque = st.slider(
                    "Torque (Nm)", 
                    min_value=0.0, 
                    max_value=80.0, 
                    value=40.0, 
                    step=0.5,
                    help="Applied torque in Newton-meters"
                )
                tool_wear = st.slider(
                    "Tool Wear (minutes)", 
                    min_value=0, 
                    max_value=260, 
                    value=100, 
                    step=1,
                    help="Accumulated tool wear time in minutes"
                )
        
        st.write("---")
        
        # model selection and predict button
        with st.container():
            st.subheader("🤖 Model Selection")
            col1, col2 = st.columns([4, 1])
            
            with col1:
                model_choice = st.selectbox(
                    "Choose Prediction Model", 
                    ["Bagging + Tomek Links (Recommended)", 
                     "Random Forest + RandomOverSampler", 
                     "Balanced Bagging"],
                    help="Select the machine learning model to use for prediction"
                )
            
            with col2:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing to align with selectbox
                predict_button = st.button(
                    "🚀 Predict Failure", 
                    type="primary", 
                    use_container_width=True
                )
        
        if predict_button:
            try:
                # Preprocess input
                input_processed = preprocess_input(
                    product_type, air_temp, process_temp, 
                    rotational_speed, torque, tool_wear
                )
                
                # Select model
                if "Bagging + Tomek Links" in model_choice:
                    model = bagging_model
                    model_name = "Bagging + Tomek Links"
                elif "Random Forest" in model_choice:
                    model = rf_model
                    model_name = "Random Forest + RandomOverSampler"
                else:
                    model = balanced_bagging_model
                    model_name = "Balanced Bagging"
                
                # Make prediction
                prediction = model.predict(input_processed)[0]
                prediction_proba = model.predict_proba(input_processed)[0]
                
                # Results section
                st.write("---")
                st.markdown(f"## 📊 Prediction Results")
                st.caption(f"Model used: **{model_name}**")
                
                # Main prediction display
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #f8d7da; border-radius: 10px; border-left: 5px solid #dc3545;'>
                            <h1 style='color: #721c24; margin: 0;'>⚠️</h1>
                            <h3 style='color: #721c24; margin: 10px 0;'>FAILURE PREDICTED</h3>
                            <p style='color: #721c24; margin: 0;'>High Risk Detected</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style='text-align: center; padding: 20px; background-color: #d4edda; border-radius: 10px; border-left: 5px solid #28a745;'>
                            <h1 style='color: #155724; margin: 0;'>✅</h1>
                            <h3 style='color: #155724; margin: 10px 0;'>NO FAILURE</h3>
                            <p style='color: #155724; margin: 0;'>Normal Operation</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style='text-align: center; padding: 10px;'>
                        <p style='font-size: 14px; color: #666; margin: 0;'>Failure Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <h1 style='color: #dc3545; margin: 5px 0; font-size: 3rem;'>{prediction_proba[1]:.1%}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown("""
                    <div style='text-align: center; padding: 10px;'>
                        <p style='font-size: 14px; color: #666; margin: 0;'>Normal Probability</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown(f"""
                    <div style='text-align: center;'>
                        <h1 style='color: #28a745; margin: 5px 0; font-size: 3rem;'>{prediction_proba[0]:.1%}</h1>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.write("")
                
                # Probability gauge and input summary
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("#### 📈 Risk Assessment Gauge")
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction_proba[1] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Failure Risk (%)", 'font': {'size': 20}},
                        delta={'reference': 3.4, 'increasing': {'color': "red"}, 'suffix': '%'},
                        number={'suffix': '%', 'font': {'size': 40}},
                        gauge={
                            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                            'bar': {'color': "darkred" if prediction == 1 else "green", 'thickness': 0.7},
                            'bgcolor': "white",
                            'borderwidth': 2,
                            'bordercolor': "gray",
                            'steps': [
                                {'range': [0, 30], 'color': '#d4edda'},
                                {'range': [30, 70], 'color': '#fff3cd'},
                                {'range': [70, 100], 'color': '#f8d7da'}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 50
                            }
                        }
                    ))
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("#### 📋 Input Summary")
                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 15px; border-radius: 8px; border: 1px solid #dee2e6;'>
                        <p style='margin: 5px 0; color: #000000;'><strong>Type:</strong> {product_type}</p>
                        <hr style='margin: 8px 0; border: none; border-top: 1px solid #dee2e6;'>
                        <p style='margin: 5px 0; color: #000000;'><strong>Air Temp:</strong> {air_temp} K</p>
                        <p style='margin: 5px 0; color: #000000;'><strong>Process Temp:</strong> {process_temp} K</p>
                        <hr style='margin: 8px 0; border: none; border-top: 1px solid #dee2e6;'>
                        <p style='margin: 5px 0; color: #000000;'><strong>RPM:</strong> {rotational_speed}</p>
                        <p style='margin: 5px 0; color: #000000;'><strong>Torque:</strong> {torque} Nm</p>
                        <p style='margin: 5px 0; color: #000000;'><strong>Tool Wear:</strong> {tool_wear} min</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Risk level indicator
                    st.write("")
                    if prediction_proba[1] >= 0.70:
                        risk_level = "🔴 Critical"
                        risk_color = "#dc3545"
                    elif prediction_proba[1] >= 0.30:
                        risk_level = "🟡 Moderate"
                        risk_color = "#ffc107"
                    elif prediction_proba[1] >= 0.15:
                        risk_level = "🟢 Low"
                        risk_color = "#28a745"
                    else:
                        risk_level = "✅ Very Low"
                        risk_color = "#20c997"
                    
                    st.markdown(f"""
                    <div style='text-align: center; padding: 10px; background-color: {risk_color}20; border-radius: 5px; border: 2px solid {risk_color};'>
                        <strong style='color: {risk_color}; font-size: 16px;'>Risk Level: {risk_level}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Recommendation section
                st.write("---")
                st.markdown("## 💡 Maintenance Recommendations")
                
                if prediction == 1:
                    with st.container():
                        st.error("""
                        ### ⚠️ IMMEDIATE ACTION REQUIRED
                        """)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("""
                            **🔴 Critical Actions (Within 24 hours):**
                            - Schedule immediate inspection
                            - Alert maintenance team
                            - Prepare necessary tools and parts
                            - Document current conditions
                            """)
                        
                        with col2:
                            st.markdown("""
                            **📊 Operational Adjustments:**
                            - Reduce operational load if possible
                            - Increase monitoring frequency
                            - Prepare for potential downtime
                            - Have backup equipment ready
                            """)
                        
                        st.info("""
                        **💰 Cost Impact:**
                        - Inspection Cost: **Rs. 5,000**
                        - Avoided Breakdown: **Rs. 500,000**
                        - **Net Savings: Rs. 495,000** ✅
                        """)
                else:
                    if prediction_proba[1] > 0.30:
                        with st.container():
                            st.warning("""
                            ### ⚡ MODERATE RISK DETECTED
                            """)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                **🟡 Recommended Actions:**
                                - Monitor closely (48-72 hours)
                                - Schedule inspection next week
                                - Review parameter trends
                                - Prepare maintenance resources
                                """)
                            
                            with col2:
                                st.markdown("""
                                **📈 Monitoring Points:**
                                - Track temperature variations
                                - Monitor RPM stability
                                - Check torque fluctuations
                                - Log tool wear progression
                                """)
                    
                    elif prediction_proba[1] > 0.15:
                        with st.container():
                            st.info("""
                            ### 🟢 LOW RISK - CONTINUE MONITORING
                            """)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                **✅ Current Status:**
                                - No immediate action needed
                                - Continue regular schedule
                                - Monitor operational parameters
                                """)
                            
                            with col2:
                                st.markdown("""
                                **📅 Next Steps:**
                                - Review during scheduled maintenance
                                - Keep logging sensor data
                                - Maintain current operating conditions
                                """)
                    
                    else:
                        with st.container():
                            st.success("""
                            ### ✅ NORMAL OPERATION
                            """)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("""
                                **🎯 Status:**
                                - All systems nominal
                                - Operating within safe parameters
                                - No concerns detected
                                """)
                            
                            with col2:
                                st.markdown("""
                                **📊 Maintenance:**
                                - Follow regular schedule
                                - Standard monitoring sufficient
                                - Continue current procedures
                                """)
                
                # Debug info 
                with st.expander("🔍 View Technical Details (Advanced)"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Processed Features:**")
                        st.dataframe(input_processed, use_container_width=True)
                    
                    with col2:
                        st.markdown("**Model Information:**")
                        st.write(f"- Model: {model_name}")
                        st.write(f"- Feature Order: {list(input_processed.columns)}")
                        st.write(f"- Input Shape: {input_processed.shape}")
                        st.write(f"- Prediction Value: {prediction}")
                        st.write(f"- Class Probabilities: {prediction_proba}")
            
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                
                with st.expander("🔧 Debug Information"):
                    st.write(f"**Error Type:** {type(e).__name__}")
                    st.write(f"**Error Details:** {str(e)}")
                    st.write(f"**Selected Model:** {model_choice}")
                    st.write(f"**Model Loaded:** {model is not None}")

# cost anyalisis page
elif page == "Cost-Benefit Analysis":
    st.header("💰 Cost-Benefit Analysis")
    
    st.write("""
    This analysis compares the financial impact of using predictive maintenance models 
    versus a traditional "Run-to-Failure" approach.
    """)
    
    st.write("---")
    
    # Cost parameters
    st.subheader("⚙️ Cost Parameters (Sri Lankan Rupees)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        cost_inspection = st.number_input("Cost of Inspection (Rs.)", 
                                         min_value=0, max_value=50000, value=5000, step=500,
                                         help="Cost of sending a technician to inspect the machine")
    
    with col2:
        cost_breakdown = st.number_input("Cost of Breakdown (Rs.)", 
                                        min_value=0, max_value=5000000, value=500000, step=10000,
                                        help="Cost of unexpected failure including downtime and repairs")
    
    st.write("---")
    
    # Model comparison data 
    # TP, FP, FN values for each model on test set of 2500 samples with 85 failures
    models_cost_data = {
        'Bagging + Tomek Links': {'TP': 60, 'FP': 9, 'FN': 25},
        'Random Forest + RandomOverSampler': {'TP': 55, 'FP': 8, 'FN': 30},
        'Balanced Bagging': {'TP': 75, 'FP': 192, 'FN': 10}
    }
    
    st.subheader("📊 Cost Comparison Results")
    
    results = []
    
    for model_name, values in models_cost_data.items():
        baseline_cost, model_cost, savings = calculate_cost_impact(
            values['TP'], values['FP'], values['FN'], 
            cost_inspection, cost_breakdown
        )
        
        results.append({
            'Model': model_name,
            'Baseline Cost (Rs.)': baseline_cost,
            'Model Cost (Rs.)': model_cost,
            'Savings (Rs.)': savings,
            'ROI (%)': (savings / baseline_cost * 100) if baseline_cost > 0 else 0,
            'True Positives': values['TP'],
            'False Positives': values['FP'],
            'False Negatives': values['FN']
        })
    
    df_results = pd.DataFrame(results)
    
    # Display results table
    st.dataframe(df_results.style.highlight_max(axis=0, subset=['Savings (Rs.)', 'ROI (%)']).format({
        'Baseline Cost (Rs.)': 'Rs.{:,.0f}',
        'Model Cost (Rs.)': 'Rs.{:,.0f}',
        'Savings (Rs.)': 'Rs.{:,.0f}',
        'ROI (%)': '{:.1f}%'
    }), use_container_width=True)
    
    st.write("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Savings comparison
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Baseline Cost',
            x=df_results['Model'],
            y=df_results['Baseline Cost (Rs.)'],
            marker_color='lightcoral'
        ))
        fig.add_trace(go.Bar(
            name='Model Cost',
            x=df_results['Model'],
            y=df_results['Model Cost (Rs.)'],
            marker_color='lightblue'
        ))
        fig.update_layout(
            title='Cost Comparison: Baseline vs Model',
            xaxis_title='Model',
            yaxis_title='Cost (Rs.)',
            barmode='group',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Savings bar chart
        fig = go.Figure(go.Bar(
            x=df_results['Model'],
            y=df_results['Savings (Rs.)'],
            text=df_results['Savings (Rs.)'].apply(lambda x: f'Rs.{x:,.0f}'),
            textposition='auto',
            marker_color=['#2ecc71', '#3498db', '#f39c12']
        ))
        fig.update_layout(
            title='Estimated Savings by Model',
            xaxis_title='Model',
            yaxis_title='Savings (Rs.)',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.write("---")
    
    # Key insights
    st.subheader("🎯 Financial Recommendations")
    
    max_savings_idx = df_results['Savings (Rs.)'].idxmax()
    max_savings_model = df_results.loc[max_savings_idx, 'Model']
    max_savings_amount = df_results.loc[max_savings_idx, 'Savings (Rs.)']
    max_roi = df_results.loc[max_savings_idx, 'ROI (%)']
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best Model (Savings)", max_savings_model.split('+')[0], 
                 f"Rs.{max_savings_amount:,.0f}")
    
    with col2:
        st.metric("Maximum ROI", f"{max_roi:.1f}%", 
                 delta="vs Run-to-Failure")
    
    with col3:
        avg_savings = df_results['Savings (Rs.)'].mean()
        st.metric("Average Savings", f"Rs.{avg_savings:,.0f}")
    
    st.write("---")
    
    # Detailed recommendations
    st.subheader("📋 Scenario-Based Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("""
        **Scenario A: Maximize Savings**
        
        **Recommended:** Balanced Bagging
        
        - Highest total savings
        - Catches 88% of failures
        - More false alarms (higher inspection costs)
        - Best if you have adequate maintenance staff
        
        **Ideal for:** Companies with internal maintenance teams
        """)
    
    with col2:
        st.info("""
        **Scenario B: Optimize Efficiency**
        
        **Recommended:** Bagging + Tomek Links
        
        - Best precision-recall balance
        - Fewer false alarms
        - Still achieves significant savings
        - More efficient resource utilization
        
        **Ideal for:** Limited maintenance resources
        """)
    
    # Cost breakdown
    st.write("---")
    st.subheader("💡 Cost Breakdown Explanation")
    
    st.info("""
    **How costs are calculated:**
    
    1. **Baseline (Run-to-Failure):** 
       - No inspections performed
       - All failures result in breakdowns
       - Cost = Total Failures × Breakdown Cost
    
    2. **With Predictive Model:**
       - Inspection Cost = (True Positives + False Positives) × Inspection Cost
       - Breakdown Cost = False Negatives × Breakdown Cost
       - Total Model Cost = Inspection Cost + Breakdown Cost
    
    3. **Savings = Baseline Cost - Model Cost**
    
    **Key Insight:** Even with false positives, predictive maintenance is significantly cheaper 
    than reactive maintenance because preventing breakdowns saves much more than the cost of 
    unnecessary inspections.
    """)

# Footer
st.write("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Predictive Maintenance Dashboard | Built with Streamlit</p>
    <p>AI4I 2020 Predictive Maintenance Dataset</p>
</div>
""", unsafe_allow_html=True)


