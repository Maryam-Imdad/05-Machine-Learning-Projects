import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="AI Grooming Tech - Maryam Imdad", layout="wide", page_icon="💄")
st.title("💄 AI Grooming Tech Analyzer - Future of Beauty")
st.markdown("**By Maryam Imdad | BS Data Science | AI + Beauty Tech**")
st.markdown("---")

@st.cache_resource
def train_grooming_model():
    np.random.seed(42)
    n=800
    data = {
        'age': np.random.randint(16,60,n),
        'oily_score': np.random.randint(1,10,n),
        'dry_score': np.random.randint(1,10,n),
        'sensitive': np.random.randint(0,2,n),
        'acne_prone': np.random.randint(0,2,n),
        'daily_sun_exposure': np.random.randint(1,10,n),
        'water_intake_liters': np.random.randint(1,5,n),
    }
    df = pd.DataFrame(data)
    def get_skin_type(row):
        if row['oily_score'] >=7: return 0
        elif row['dry_score'] >=7: return 1
        elif row['oily_score']>=5 and row['dry_score']>=5: return 2
        else: return 3
    df['skin_type'] = df.apply(get_skin_type, axis=1)
    X = df.drop('skin_type', axis=1)
    y = df['skin_type']
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X,y)
    return model, df

model, df_train = train_grooming_model()
skin_labels = {0:"Oily Skin", 1:"Dry Skin", 2:"Combination Skin", 3:"Normal Skin"}

col1, col2 = st.columns([1,1.2])
with col1:
    st.subheader("✨ Enter Your Details")
    age = st.slider("Age", 16, 60, 24)
    oily = st.slider("How Oily does skin feel? (1-10)", 1,10,5)
    dry = st.slider("How Dry does skin feel? (1-10)", 1,10,5)
    sensitive = st.selectbox("Sensitive Skin?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    acne = st.selectbox("Acne Prone?", [0,1], format_func=lambda x: "Yes" if x==1 else "No")
    sun = st.slider("Daily Sun Exposure (1-10)", 1,10,5)
    water = st.slider("Water Intake L/day", 1,5,2)
    gender = st.selectbox("Gender", ["Female","Male"])
    btn = st.button("💅 Analyze & Get Routine", type="primary", use_container_width=True)

with col2:
    if btn:
        inp = [[age, oily, dry, sensitive, acne, sun, water]]
        pred = model.predict(inp)[0]
        prob = max(model.predict_proba(inp)[0])
        st.subheader("🔬 AI Analysis Result")
        st.success(f"### Your Skin Type: {skin_labels[pred]}")
        st.metric("Confidence", f"{prob*100:.1f}%")
        st.subheader("🧴 Personalized Routine + New Tech")
        if pred == 0:
            st.write("**Morning:** Gel Cleanser + Niacinamide + Oil-Free Moisturizer + SPF 50")
            st.write("**Tech:** AI Oil Control Device - Smart Sebum Analyzer, Smart Mirror")
        elif pred == 1:
            st.write("**Morning:** Cream Cleanser + Hyaluronic Acid + Rich Moisturizer + SPF 30")
            st.write("**Tech:** Hydra Sensor - Skin Hydration Tracker, LED Mask")
        elif pred == 2:
            st.write("**Morning:** Foaming Cleanser + Vitamin C + Gel Moisturizer + SPF")
            st.write("**Tech:** Smart Mirror - Skin Mapper, Ultrasonic Scrubber")
        else:
            st.write("**Morning:** Gentle Cleanser + Vitamin C + Moisturizer + SPF")
            st.write("**Tech:** AI Beauty Mirror - Daily Skin Score")
        if gender == "Female":
            st.info("💡 New Tech: LED Face Mask, Ultrasonic Scrubber, AI Hair Analyzer")
        else:
            st.info("💡 New Tech: Smart Beard Trimmer, AI Scalp Analyzer, Laser Device")
        fig = go.Figure(go.Indicator(mode="gauge+number", value=prob*100, title={'text': "Skin Health Score"}, gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "#ff69b4"}}))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👈 Enter details to get analysis")
        fig = px.pie(df_train, names=df_train['skin_type'].map(skin_labels), title="Skin Types Distribution")
        st.plotly_chart(fig, use_container_width=True)

st.caption("Made with ❤️ by Maryam Imdad | AI + Beauty Tech")