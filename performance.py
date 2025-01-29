import pickle
import streamlit as st
import pandas as pd
import numpy as np

# Set up Streamlit page config
st.set_page_config(page_title="Student Performance Prediction", page_icon="⌛", layout="wide")

# Custom CSS to set background color to sky blue
page_bg_color = """
<style>
    body {
        background-color: #87CEEB !important; /* Sky Blue */
    }
    .stApp {
        background-color: #87CEEB !important;
    }
</style>
"""
st.markdown(page_bg_color, unsafe_allow_html=True)

# Load the prediction model
model = pickle.load(open('model.pkl', 'rb'))

st.title('STUDENT PERFORMANCE PREDICTION ⌛')
st.subheader('Enter Your Data 📊')

# Input fields for user data
Hours_Studied = st.number_input('Enter Hours Studied')
Attendance = st.number_input('Enter the Attendance of the Student')
Access_to_Resources_m = st.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
Motivation_Level_m = st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])

# Prepare input data
input_data = {
    'Hours_Studied': Hours_Studied,
    'Attendance': Attendance,
    'Access_to_Resources_m': Access_to_Resources_m,
    'Motivation_Level_m': Motivation_Level_m
}

# Convert input to a DataFrame
new_data = pd.DataFrame([input_data])

# Preprocess the data
df = pd.read_csv('student_performance_preprocessed_data.csv')
columns_list = df.columns.to_list()

# Map categorical values to numerical ones
new_data['Access_to_Resources_m'] = new_data['Access_to_Resources_m'].map({'Low': 1, 'Medium': 2, 'High': 3})
new_data['Motivation_Level_m'] = new_data['Motivation_Level_m'].map({'Low': 1, 'Medium': 2, 'High': 3})

# Reindex the new data to match the model's expected format
df_preprocessed = new_data.reindex(columns=columns_list, fill_value=0)

# Prediction
if st.button('PREDICT'):
    prediction = model.predict(df_preprocessed)
    
    st.write(f'The Predicted Score Is: `{prediction[0]}`')
    if prediction[0]>40:
        print('passed')
        st.balloons()