import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

@st.cache_data
def load_model():
    # Load and train the model
    df = pd.read_csv('score.csv')
    X = df[['Hours']]
    y = df['Scores']
    model = LinearRegression().fit(X, y)
    return model

model = load_model()

# App title and description
st.title("Study Hours → Predicted Score")
st.write("Input your study hours to see your predicted test score.")

# User input
hours = st.slider("Hours Studied", 0.0, 12.0, 5.0, step=0.25)

# Prediction
if st.button("Predict"):
    df_new = pd.DataFrame({'Hours': [hours]})
    pred = model.predict(df_new)[0]
    st.success(f"Predicted Score: {pred:.2f}")

# Optionally show raw data
if st.checkbox("Show raw data"):
    df = pd.read_csv('score.csv')
    st.dataframe(df)
