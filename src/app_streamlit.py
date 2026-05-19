import sys, os
_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _root not in sys.path: sys.path.insert(0, _root)
import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
try:
    from src.main import main as run_training_logic
    from src.run_many_seeds import run_multiple_seeds
    from src.data_loading import load_df, define_columns
    from src.config import MLFLOW_TRACKING_URI
    _TRAINING_AVAILABLE = True
except ImportError as e:
    _TRAINING_AVAILABLE = False
    _TRAINING_ERROR = str(e)

# Page configuration
st.set_page_config(
    page_title="Cold Plasma Seed Treatment Framework",
    page_icon="🌱",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    .stTabs [aria-selected="true"] { background-color: #10b981; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with Paper Reference
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/microscope.png", width=80)
    st.markdown("### 📚 Research Reference")
    st.markdown("""
    | Item | Detail |
    | :--- | :--- |
    | **Paper** | Cold plasma seed treatment effect on germination rate: A machine learning framework integrating seed traits and plasma parameters for predicting germination uplift in crops |
    | **Link** | [arXiv:2510.23657](https://arxiv.org/abs/2510.23657) |
    """)
    st.divider()
    
    # MLflow Dashboard Link (Only visible in Docker/Local)
    if not os.environ.get("STREAMLIT_RUNTIME_ENV"):
        st.markdown("### 📊 Model Supervision")
        st.info("Monitoring server detected. Access the dashboard to compare experiments and visualize training metrics.")
        st.markdown("[🚀 Open MLflow Dashboard](http://localhost:5000)")
        st.divider()

    st.info("This interface implements the ML pipeline described in the paper for predicting seed germination performance after cold plasma treatment.")

st.title("🌱 Cold Plasma Seed Treatment: Germination Prediction")
st.markdown("#### A machine learning framework integrating seed traits and plasma parameters for predicting germination uplift in crops")

tab1, tab2 = st.tabs(["🚀 Real-Time Inference", "🧪 Experimentation & MLOps"])

with tab1:
    st.header("Predict Germination Performance")
    
    # Priority: 1. Repo root (good model) 2. /app/outputs 3. /tmp (session training)
    _repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_paths = [
        os.path.join(_repo_root, "outputs", "model.pkl"),
        "/app/outputs/model.pkl",
        "outputs/model.pkl",
        "/tmp/outputs/model.pkl"
    ]
    
    model_path = next((p for p in model_paths if os.path.exists(p)), None)
    
    if model_path:
        st.success(f"✅ Active Model Ready.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📦 Seed Traits")
            base_germ = st.slider("Base Germination Rate (%)", 0.0, 100.0, 75.0)
            base_idx = st.number_input("Base Germination Index", 0.0, 100.0, 12.0)
            base_pot = st.number_input("Base Germination Potential (%)", 0.0, 100.0, 80.0)
            seed_weight = st.number_input("Weight of each seed (gr)", 0.0, 1.0, 0.05, format="%.3f")
            seed_size = st.number_input("Size of each seed (mm)", 0.0, 20.0, 5.0)
            sod = st.number_input("Baseline SOD (u g-1)", 0.0, 100.0, 1.5)

        with col2:
            st.subheader("⚡ Plasma Parameters")
            voltage = st.slider("Voltage (kV)", 0.0, 50.0, 15.0)
            power = st.number_input("Power (w)", 0.0, 500.0, 25.0)
            p_time = st.number_input("Plasma Time (s)", 0, 3600, 300)
            gas = st.selectbox("Gas Type", ["Air", "O2", "N2", "Ar"])

        with col3:
            st.subheader("🌡️ Conditions")
            germ_days = st.number_input("Germination Days", 1, 30, 7)

        if st.button("Run Prediction", type="primary"):
            # Construct DataFrame with EXACT naming and ORDERING from training
            input_dict = {
                'size of each seed (mm)': seed_size,
                'weight of each seed (gr)': seed_weight,
                'baseline SOD (u g-1)': sod,
                'base germination rate': base_germ,
                'base germination potential': base_pot,
                'base germination index': base_idx,
                'voltage (kV)': voltage,
                'power (w)': power,
                'plasma time': p_time,
                'germination days': germ_days,
                'gas': gas
            }
            
            try:
                # Load training config to ensure column match if possible
                if _TRAINING_AVAILABLE:
                    df_sample = load_df()
                    f_cols, _, _, _ = define_columns(df_sample)
                    input_data = pd.DataFrame([input_dict])[f_cols]
                else:
                    input_data = pd.DataFrame([input_dict])

                model = joblib.load(model_path)
                prediction = model.predict(input_data)[0]
                
                # Sanity check: Germination rate can't be > 100% or significantly < base
                prediction = max(0, min(100, prediction))
                
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                with res_col1: 
                    st.metric("Predicted Final Rate", f"{prediction:.2f}%")
                with res_col2: 
                    uplift = prediction - base_germ
                    st.metric("Estimated Uplift", f"{uplift:+.2f}%", delta_color="normal")
                
                if uplift > 0:
                    st.balloons()
                    st.success(f"Targeted treatment predicted to increase germination by {uplift:.1f}%")
                else:
                    st.warning("Selected parameters may not provide significant germination uplift.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Tip: This often happens if the input columns don't match the trained model features.")
    else:
        st.warning("⚠️ No pre-trained model found. Please run the training pipeline in the MLOps tab first.")

with tab2:
    st.header("MLOps Laboratory")
    if not _TRAINING_AVAILABLE:
        st.error(f"Training modules unavailable: {_TRAINING_ERROR}")
        st.stop()

    mode = st.radio("Experiment Mode", ["Single Model Tuning", "Full Pipeline Comparison", "Robustness Test (Many Seeds)"], horizontal=True)
    col_a, col_b = st.columns([2, 1])

    with col_a:
        if mode == "Single Model Tuning":
            model_choice = st.selectbox("Select Model Architecture", ["RF", "ET", "GB", "XGB"])
            custom_params = {}
            if model_choice == "XGB":
                lr = st.select_slider("learning_rate", options=[0.01, 0.05, 0.1], value=0.05)
                max_d = st.slider("max_depth", 3, 12, 6)
                custom_params = {"model__learning_rate": [lr], "model__max_depth": [max_d]}
            
            rand_st = st.number_input("Random State", 1, 1000, 42)
            if st.button("🚀 Train Specific Model"):
                with st.spinner(f"Training {model_choice}..."):
                    run_training_logic(random_state=rand_st, selected_model_name=model_choice, custom_params=custom_params)
                    st.success("Training Complete! Refresh page to use new model.")

        elif mode == "Full Pipeline Comparison":
            if st.button("🏁 Run All Models"):
                with st.spinner("Evaluating all architectures..."):
                    results = run_training_logic()
                    st.table(results.drop(columns=["best_pipe"], errors="ignore"))

    with col_b:
        st.subheader("Training Logs")
        st.write(f"Tracking URI: `{MLFLOW_TRACKING_URI}`")
        if "localhost" in MLFLOW_TRACKING_URI or "mlflow-server" in MLFLOW_TRACKING_URI:
             st.success("Connected to MLflow Tracking Server")
        else:
             st.info("Using local filesystem for tracking")
