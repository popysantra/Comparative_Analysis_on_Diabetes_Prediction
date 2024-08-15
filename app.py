import streamlit as st
import pickle
import pandas as pd

# Load all models
with open('logistic_regression_model.pkl', 'rb') as file:
    lr_model = pickle.load(file)

with open('decision_tree_model.pkl', 'rb') as file:
    dt_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    rf_model = pickle.load(file)

# Set up the Streamlit app
st.title('Diabetes Prediction Using Machine Learning Models')

# Input fields for the user
pregnancies = st.number_input('Pregnancies', min_value=0, value=0)
glucose = st.number_input('Glucose', min_value=0, value=0)
blood_pressure = st.number_input('Blood Pressure', min_value=0, value=0)
skin_thickness = st.number_input('Skin Thickness', min_value=0, value=0)
insulin = st.number_input('Insulin', min_value=0, value=0)
bmi = st.number_input('BMI', min_value=0.0, value=0.0)
diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function', min_value=0.0, value=0.0)
age = st.number_input('Age', min_value=0, value=0)

# Create a DataFrame for prediction
data = {
    'Pregnancies': [pregnancies],
    'Glucose': [glucose],
    'BloodPressure': [blood_pressure],
    'SkinThickness': [skin_thickness],
    'Insulin': [insulin],
    'BMI': [bmi],
    'DiabetesPedigreeFunction': [diabetes_pedigree_function],
    'Age': [age]
}
df = pd.DataFrame(data)

# Model selection and prediction
model_selection = st.radio("Select Model", ("Logistic Regression", "Decision Tree", "Random Forest"))

if model_selection == "Logistic Regression":
    model = lr_model
elif model_selection == "Decision Tree":
    model = dt_model
elif model_selection == "Random Forest":
    model = rf_model

if st.button('Predict'):
    prediction = model.predict(df)
    if prediction[0] == 1:
        st.write('Prediction: Positive')
        st.write('Suggestions:')
        st.write('- Regularly monitor blood glucose levels.')
        st.write('- Follow a healthy diet with low sugar intake.')
        st.write('- Engage in regular physical activity.')
        st.write('- Consult a healthcare provider for personalized advice.')
    else:
        st.write('Prediction: Negative')
