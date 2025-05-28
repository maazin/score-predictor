#📊 Study Hours → Score Predictor

Welcome to the Study Hours → Score Predictor web app! This simple, interactive tool lets you explore how the number of hours you study translates into an expected exam score—powered by a linear regression model trained on real historical data.

##Live Demo


👉 Try it right now at: https://share.streamlit.io/maazin/score-predictor/main/app.py

(No sign‑up required! Just move the slider and click “Predict Score.”)


 ## How to Use
 
 Adjust the slider under “Select hours studied” to choose a value between the minimum and maximum study hours in the dataset.
 
 Click the Predict Score button.

View your predicted exam score displayed in real time.

Explore the scatter plot—each point is a real student’s hours vs. score, and the orange line is the fitted regression.

That’s it! It’s designed to be intuitive, responsive, and educational.


## What’s Behind the App?

Data: A CSV (score.csv) of past students’ study hours and exam scores.

Model: A Simple Linear Regression trained with scikit-learn.

UI: Built using Streamlit, so everything runs in your browser.


## Why This Matters

Visual Learning: See at a glance how study time correlates with performance.

Quick Predictions: Estimate your own expected score to plan study sessions.

Hands‑on Example: A beginner‑friendly demonstration of data science and web apps.

