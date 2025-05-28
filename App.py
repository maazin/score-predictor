import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

@st.cache_data
# Load and cache the data
def load_data(path='score.csv'):
    df = pd.read_csv(path)
    return df

@st.cache_data
# Train and cache the linear regression model
def train_model(df):
    X = df[['Hours']]
    y = df['Scores']
    model = LinearRegression()
    model.fit(X, y)
    return model

# Plot the data and regression line
def plot_regression(df, model):
    fig, ax = plt.subplots()
    ax.scatter(df['Hours'], df['Scores'], alpha=0.7)
    ax.plot(df['Hours'], model.predict(df[['Hours']]), linewidth=2)
    ax.set_xlabel('Hours Studied')
    ax.set_ylabel('Scores')
    ax.set_title('Study Hours vs. Scores')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

# Main app function
def main():
    st.set_page_config(page_title="Score Predictor", layout="centered")
    st.title("📊 Study Hours → Score Predictor")

    # Load data and model
    df = load_data()
    model = train_model(df)

    # Show regression equation
    coef = model.coef_[0]
    intercept = model.intercept_
    st.markdown(f"**Model:** Score = {intercept:.2f} + {coef:.2f} × Hours")

    # User input
    hours = st.slider("Select hours studied:", min_value=float(df['Hours'].min()),
                      max_value=float(df['Hours'].max()), value=1.0, step=0.25)

    # Prediction
    if st.button("Predict Score"):
        X_new = pd.DataFrame({'Hours': [hours]})
        pred = model.predict(X_new)[0]
        st.success(f"Predicted Score: {pred:.2f}")

    # Show plot
    st.pyplot(plot_regression(df, model))

if __name__ == "__main__":
    main()
