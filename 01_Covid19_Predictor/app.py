import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px

st.set_page_config(page_title="COVID-19 Predictor - Maryam Imdad", layout="wide", page_icon="🦠")
st.title("🦠 COVID-19 Risk Predictor - Corona Virus Detection")
st.markdown("**By Maryam Imdad | BS Data Science | Gujranwala**")
st.markdown("---")

@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 1000
    data = {
        'fever': np.random.randint(0,2,n),
        'cough': np.random.randint(0,2,n),
        'fatigue': np.random.randint(0,2,n),
        'loss_of_taste': np.random.randint(0,2,n),
        'breathing_difficulty': np.random.randint(0,2,n),
        'age': np.random.randint(18,80,n),
        'travel_history': np.random.randint(0,2,n),
        'contact_positive': np.random.randint(0,2,n),
    }
    df = pd.DataFrame(data)
    df['covid_positive'] = ((df['fever']+df['cough']+df['loss_of_taste']+df['contact_positive'] >=2) & (np.random.rand(n) > 0.2)).astype(int)
    X = df.drop('covid_positive', axis=1)
    y = df['covid_positive']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df

model, train_df = train_model()

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("🩺 Enter Symptoms")
    fever = st.selectbox("Fever?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    cough = st.selectbox("Dry Cough?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    fatigue = st.selectbox("Fatigue?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    loss_taste = st.selectbox("Loss of Taste/Smell?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    breathing = st.selectbox("Breathing Difficulty?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    age = st.slider("Age", 18, 80, 30)
    travel = st.selectbox("Recent Travel History?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    contact = st.selectbox("Contact with COVID+ Person?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    predict_btn = st.button("🔍 Predict COVID Risk", type="primary", use_container_width=True)

with col2:
    if predict_btn:
        input_data = [[fever, cough, fatigue, loss_taste, breathing, age, travel, contact]]
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]
        st.subheader("📊 Prediction Result")
        if pred == 1:
            st.error(f"### ⚠️ HIGH RISK - COVID Positive Chance: {prob*100:.1f}%")
            st.warning("Recommendation: Isolate immediately, Consult Doctor, Get PCR Test done. Wear Mask.")
        else:
            st.success(f"### ✅ LOW RISK - COVID Positive Chance: {prob*100:.1f}%")
            st.info("Recommendation: Follow SOPs, monitor symptoms for 5 days.")
        fig = px.pie(values=[prob, 1-prob], names=['Risk','Safe'], hole=0.6, color_discrete_sequence=['#ff4b4b','#00cc96'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Enter symptoms and click Predict")
        fig2 = px.histogram(train_df, x='age', color='covid_positive', barmode='overlay', title="Age vs COVID Cases in Training Data")
        st.plotly_chart(fig2, use_container_width=True)

st.caption("Made with ❤️ by Maryam Imdad | Educational ML Model")