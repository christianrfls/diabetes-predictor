import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the Model
@st.cache_resource
def load_model():
    try:
        return joblib.load('random_forest_diabetes_predictor_model_v2.pkl')
    except FileNotFoundError:
        return None

model = load_model()

# 2. App Configuration
st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")
st.title("🩺 Diabetes Prediction Tool")
st.write("Enter patient metrics below or use the buttons to load example profiles.")

if model is None:
    st.error("Model file 'random_forest_diabetes_predictor_model.pkl' not found. Please upload it.")
else:
    # Auto-fill
    
    # Initialize session state keys if they don't exist yet
    default_values = {
        'pregnancies': 0, 'glucose': 100, 'bp': 70, 'skin': 20,
        'insulin': 0, 'bmi': 30.0, 'dpf': 0.5, 'age': 30
    }
    for key, val in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # Define the profiles
    col_demo1, col_demo2 = st.columns(2)
    
    with col_demo1:
        if st.button("Load 'Healthy' Profile 🟢"):
            st.session_state['pregnancies'] = 1
            st.session_state['glucose'] = 88
            st.session_state['bp'] = 72
            st.session_state['skin'] = 20
            st.session_state['insulin'] = 85
            st.session_state['bmi'] = 26.6
            st.session_state['dpf'] = 0.35
            st.session_state['age'] = 31
            st.rerun() # Force reload to update fields

    with col_demo2:
        if st.button("Load 'At-Risk' Profile 🔴"):
            st.session_state['pregnancies'] = 4
            st.session_state['glucose'] = 162
            st.session_state['bp'] = 88
            st.session_state['skin'] = 35
            st.session_state['insulin'] = 209
            st.session_state['bmi'] = 33.8
            st.session_state['dpf'] = 0.67
            st.session_state['age'] = 47
            st.rerun() # Force reload to update fields

    # Input Fields
    st.write("---")
    col1, col2 = st.columns(2)

    with col1:
        # Note the use of key=... this links the widget to the button logic above
        pregnancies = st.number_input("Pregnancies (No. of times)", 0, 20, key='pregnancies')
        glucose = st.number_input("Glucose (mg/dL)", 0, 200, key='glucose')
        blood_pressure = st.number_input("Diastolic Blood Pressure (mmHg)", 0, 140, key='bp')
        skin_thickness = st.number_input("Triceps Skin Fold Thickness (mm)", 0, 100, key='skin')

    with col2:
        insulin = st.number_input("Insulin (mµU/mL)", 0, 900, key='insulin')
        bmi = st.number_input("BMI (kg/m^2)", 0.0, 70.0, step=0.1, key='bmi')
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, step=0.01, key='dpf')
        age = st.number_input("Age (years)", 0, 100, key='age')

    # PROCESS AND PREDICT
    st.write("---")
    if st.button("Predict Risk", type="primary"):
        
        # A. Create DataFrame from raw inputs
        input_data = {
            'Pregnancies': [pregnancies],
            'Glucose': [glucose],
            'BloodPressure': [blood_pressure],
            'SkinThickness': [skin_thickness],
            'Insulin': [insulin],
            'BMI': [bmi],
            'DiabetesPedigreeFunction': [dpf],
            'Age': [age]
        }
        df = pd.DataFrame(input_data)

        # B. Feature Engineering (Matches the training model)
        # N1: Age <= 30 and Glucose <= 120
        df['N1'] = 0
        df.loc[(df['Age'] <= 30) & (df['Glucose'] <= 120), 'N1'] = 1
        
        # N2: BMI <= 30
        df['N2'] = 0
        df.loc[(df['BMI'] <= 30), 'N2'] = 1

        # N3: Age <= 30 and Pregnancies <= 6
        df['N3'] = 0
        df.loc[(df['Age'] <= 30) & (df['Pregnancies'] <= 6), 'N3'] = 1

        # N4: Glucose <= 105 and BloodPressure <= 80
        df['N4'] = 0
        df.loc[(df['Glucose'] <= 105) & (df['BloodPressure'] <= 80), 'N4'] = 1

        # N5: SkinThickness <= 20
        df['N5'] = 0
        df.loc[(df['SkinThickness'] <= 20), 'N5'] = 1

        # N6: BMI < 30 and SkinThickness <= 20
        df['N6'] = 0
        df.loc[(df['BMI'] < 30) & (df['SkinThickness'] <= 20), 'N6'] = 1

        # N7: Glucose <= 105 and BMI <= 30
        df['N7'] = 0
        df.loc[(df['Glucose'] <= 105) & (df['BMI'] <= 30), 'N7'] = 1

        # N8: Insulin < 150
        df['N8'] = 0
        df.loc[(df['Insulin'] < 150), 'N8'] = 1

        # N9: BloodPressure < 80
        df['N9'] = 0
        df.loc[(df['BloodPressure'] < 80), 'N9'] = 1

        # N10: Pregnancies < 4 and != 0
        df['N10'] = 0
        df.loc[(df['Pregnancies'] < 4) & (df['Pregnancies'] != 0), 'N10'] = 1

        # N0: BMI * SkinThickness
        df['N0'] = df['BMI'] * df['SkinThickness']

        # N11: Pregnancies / Age
        df['N11'] = df['Pregnancies'] / df['Age']

        # N12: Glucose / DPF
        df['N12'] = df['Glucose'] / df['DiabetesPedigreeFunction']

        # N13: Age * DPF
        df['N13'] = df['Age'] * df['DiabetesPedigreeFunction']

        # N14: Age / Insulin (Handle division by zero)
        df['N14'] = df['Age'] / df['Insulin']
        df['N14'] = df['N14'].replace([np.inf, -np.inf], 0).fillna(0)

        # N15: N0 < 1034
        df['N15'] = 0
        df.loc[(df['N0'] < 1034), 'N15'] = 1

        # C. Column Alignment
        if hasattr(model, 'feature_names_in_'):
            expected_cols = model.feature_names_in_
            df_final = df.reindex(columns=expected_cols, fill_value=0)
        else:
            df_final = df

        # D. Prediction
        prediction = model.predict(df_final)[0]
        probability = model.predict_proba(df_final)[0][1]

        # E. Output
        if prediction == 1:
            st.error(f"**Result: Diabetic Risk**")
            st.write(f"Confidence: **{probability*100:.2f}%**")
            st.markdown("""
                <style>
                    .stProgress > div > div > div > div {
                        background-color: red;
                    }
                </style>
                """, unsafe_allow_html=True)
            st.progress(int(probability*100))
        else:
            st.success(f"**Result: Healthy**")
            st.write(f"Confidence: **{(1-probability)*100:.2f}%**")
            st.markdown("""
                <style>
                    .stProgress > div > div > div > div {
                        background-color: green;
                    }
                </style>
                """, unsafe_allow_html=True)
            st.progress(int((1-probability)*100))