import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
from src.main import main as run_training_logic
from src.run_many_seeds import run_multiple_seeds

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
    st.info("This interface implements the ML pipeline described in the paper for predicting seed germination performance after cold plasma treatment.")

st.title("🌱 Cold Plasma Seed Treatment: Germination Prediction")
st.markdown("#### A machine learning framework integrating seed traits and plasma parameters for predicting germination uplift in crops")

tab1, tab2 = st.tabs(["🚀 Real-Time Inference", "🧪 Experimentation & MLOps"])

with tab1:
    st.header("Predict Germination Performance")
    # Check both absolute (Docker) and relative (Local/Cloud) paths
    model_path = "outputs/model.pkl"
    if not os.path.exists(model_path):
        model_path = "/app/outputs/model.pkl"
    
    if os.path.exists(model_path):
        st.success("✅ Active Model Ready.")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.subheader("📦 Seed Traits")
            base_germ = st.slider("Base Germination Rate", 0.0, 100.0, 75.0)
            base_idx = st.number_input("Base Germination Index", 0.0, 50.0, 12.0)
            base_pot = st.number_input("Base Germination Potential", 0.0, 100.0, 80.0)
            seed_weight = st.number_input("Weight of each seed (gr)", 0.0, 1.0, 0.05, format="%.3f")
            seed_size = st.number_input("Size of each seed (mm)", 0.0, 20.0, 5.0)
            sod = st.number_input("Baseline SOD (u g-1)", 0.0, 10.0, 1.5)

        with col2:
            st.subheader("⚡ Plasma Parameters")
            voltage = st.slider("Voltage (kV)", 0.0, 30.0, 15.0)
            power = st.number_input("Power (w)", 0.0, 100.0, 25.0)
            p_time = st.number_input("Plasma Time (s)", 0, 1000, 300)
            gas = st.selectbox("Gas Type", ["Air", "O2", "N2", "Ar"])

        with col3:
            st.subheader("🌡️ Conditions")
            germ_days = st.number_input("Germination Days", 1, 30, 7)

        if st.button("Run Prediction", type="primary"):
            input_data = pd.DataFrame([{
                'size of each seed (mm)': seed_size, 'weight of each seed (gr)': seed_weight,
                'baseline SOD (u g-1)': sod, 'base germination rate': base_germ,
                'base germination potential': base_pot, 'base germination index': base_idx,
                'voltage (kV)': voltage, 'power (w)': power, 'plasma time': p_time,
                'germination days': germ_days, 'gas': gas
            }])

            try:
                model = joblib.load(model_path)
                prediction = model.predict(input_data)[0]
                st.markdown("---")
                res_col1, res_col2 = st.columns(2)
                with res_col1: st.metric("Predicted Final Rate", f"{prediction:.2f}%")
                with res_col2: st.metric("Estimated Uplift", f"{prediction - base_germ:+.2f}%")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("⚠️ No model found. Run training first.")

with tab2:
    st.header("MLOps Laboratory")
    
    mode = st.radio("Experiment Mode", ["Single Model Tuning", "Full Pipeline Comparison", "Robustness Test (Many Seeds)"], horizontal=True)
    
    col_a, col_b = st.columns([2, 1])
    
    with col_a:
        if mode == "Single Model Tuning":
            model_choice = st.selectbox("Select Model Architecture", ["ET", "GB", "XGB"])
            custom_params = {}
            if model_choice == "ET":
                n_est = st.slider("n_estimators", 100, 1000, 500)
                max_dep = st.slider("max_depth", 1, 30, 8)
                custom_params = {"model__n_estimators": [n_est], "model__max_depth": [max_dep]}
            elif model_choice == "GB":
                lr = st.select_slider("learning_rate", options=[0.001, 0.01, 0.1], value=0.01)
                n_est = st.slider("n_estimators", 100, 1000, 400)
                custom_params = {"model__learning_rate": [lr], "model__n_estimators": [n_est]}
            elif model_choice == "XGB":
                lr = st.select_slider("learning_rate", options=[0.01, 0.05, 0.1], value=0.05)
                gamma = st.slider("gamma", 0, 20, 10)
                custom_params = {"model__learning_rate": [lr], "model__gamma": [gamma]}
            
            rand_st = st.number_input("Random State", 1, 1000, 42)
            if st.button("🚀 Train Specific Model"):
                with st.spinner(f"Training {model_choice}..."):
                    run_training_logic(random_state=rand_st, selected_model_name=model_choice, custom_params=custom_params)
                    st.success("Done!")

        elif mode == "Full Pipeline Comparison":
            st.info("This will benchmark all base models and hybrid stacking variants.")
            rand_st = st.number_input("Random State", 1, 1000, 42)
            if st.button("🚀 Run Comparison Pipeline"):
                with st.spinner("Executing ET, GB, XGB, and 4 Hybrid models..."):
                    run_training_logic(random_state=rand_st)
                    st.success("Pipeline comparison complete!")

        elif mode == "Robustness Test (Many Seeds)":
            st.warning("Warning: This involves multiple training cycles and may take time.")
            seeds_input = st.text_input("Enter Seeds (comma separated)", "42, 123, 456")
            seeds = [int(s.strip()) for s in seeds_input.split(",")]
            if st.button("🚀 Run Seed Robustness Analysis"):
                with st.spinner(f"Running pipeline across {len(seeds)} different seeds..."):
                    combined, stats = run_multiple_seeds(seeds=seeds)
                    st.table(stats[['model', 'test_R2_mean', 'test_R2_std']])
                    st.success("Robustness stats logged to MLflow!")

    with col_b:
        st.subheader("MLOps Dashboard")
        st.markdown(f"""
            <a href="http://localhost:5000" target="_blank">
                <button style="width:100%; height:60px; background-color:#0ea5e9; color:white; border:none; border-radius:10px; font-weight:bold; cursor:pointer;">
                    📊 Open MLflow UI
                </button>
            </a>
            """, unsafe_allow_html=True)
        st.info("Check MLflow for comparison charts, aggregated statistics, and model residuals.")

st.markdown("---")
st.caption("Developed for Saklain Niam Portfolio - Enterprise MLOps Implementation")
