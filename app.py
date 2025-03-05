import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import os

def set_page_style():
    st.markdown("""
        <style>
             .stApp { background-color: #acacd7; }
            .big-font { font-size:20px !important; font-weight: bold; }
            .stButton>button { background-color: #4CAF50; color: white; border-radius: 10px; }
            .stProgress>div>div { background-color: #4CAF50; }
        </style>
    """, unsafe_allow_html=True)

set_page_style()

# Page Title
st.title("🏋️ Personal Fitness Tracker")
st.write("Track your exercise metrics and predict calories burned based on various inputs!")

# Sidebar Input Fields
st.sidebar.header("📊 User Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min)", 0, 60, 15)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 80)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender_button = st.sidebar.radio("Gender", ("Male", "Female"))
    gender = 1 if gender_button == "Male" else 0
    return pd.DataFrame({"Age": [age], "BMI": [bmi], "Duration": [duration], "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [gender]})

user_data = user_input_features()
st.write("### 🏃 Your Input Parameters")
st.dataframe(user_data, width=600)

# Load and preprocess dataset
base_path = os.path.dirname(os.path.abspath(__file__))
calories_path = os.path.join(base_path, "calories.csv")
exercise_path = os.path.join(base_path, "exercise.csv")

calories = pd.read_csv(calories_path)
exercise = pd.read_csv(exercise_path)

exercise_df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
exercise_df["BMI"] = round(exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2), 2)
exercise_df = pd.get_dummies(exercise_df, drop_first=True)

# Split data
X = exercise_df.drop(columns=["Calories"])
y = exercise_df["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# Model Selection
st.sidebar.header("🧠 Choose Model")
model_choice = st.sidebar.radio("Select a model", ["Random Forest", "Linear Regression"])

if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=500, max_features=3, max_depth=6)
else:
    model = LinearRegression()

# Train Model
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Accuracy Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.write(f"### 📈 Model Performance: {model_choice}")
st.write(f"**Mean Squared Error:** {mse:.2f}")
st.write(f"**R² Score:** {r2:.2f}")

# Make Prediction
user_data = user_data.reindex(columns=X_train.columns, fill_value=0)
prediction = model.predict(user_data)[0]
st.success(f"🔥 Predicted Calories Burned: {round(prediction, 2)} kcal")

# Data Visualization
st.write("### 📊 Data Insights")
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(exercise_df["Calories"], bins=30, kde=True, color='blue', alpha=0.6, ax=ax)
ax.set_title("Distribution of Calories Burned")
st.pyplot(fig)

# Correlation Heatmap
st.write("### 🔥 Correlation Between Features")
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(exercise_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
st.pyplot(fig)

# Additional Feature: Activity Recommendation
st.write("### 🏆 Personalized Activity Recommendation")
if prediction < 200:
    st.info("🔵 Light exercise recommended: Walking, Stretching")
elif 200 <= prediction < 400:
    st.info("🟢 Moderate exercise recommended: Jogging, Yoga")
else:
    st.info("🔴 Intense exercise recommended: Running, HIIT, Weight Training")

# General Information Comparison
st.write("### 📋 General Information")
st.write(f"You are older than {round((exercise_df['Age'] < user_data['Age'].values[0]).mean() * 100, 2)}% of other people.")
st.write(f"Your exercise duration is higher than {round((exercise_df['Duration'] < user_data['Duration'].values[0]).mean() * 100, 2)}% of other people.")
st.write(f"You have a higher heart rate than {round((exercise_df['Heart_Rate'] < user_data['Heart_Rate'].values[0]).mean() * 100, 2)}% of other people during exercise.")
st.write(f"You have a higher body temperature than {round((exercise_df['Body_Temp'] < user_data['Body_Temp'].values[0]).mean() * 100, 2)}% of other people during exercise.")
