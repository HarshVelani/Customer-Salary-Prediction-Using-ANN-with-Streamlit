import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle


# Load the trained model
model = tf.keras.models.load_model('Customer_Salary_Prediction\salary_regression_model.h5')


# Load the encoder and scaler
with open('Customer_Salary_Prediction\Reg_label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('Customer_Salary_Prediction\Reg_onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('Customer_Salary_Prediction\Reg_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)


# Streamlit app
st.title('Customer Salary Prediction')

# User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
# estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
exited = st.selectbox('Exited', [0, 1]) 

# Prepare input data

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    # 'Geography': [onehot_encoder_geo.transform([geography])[0]],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [exited]
    # 'EstimatedSalary': [estimated_salary]
})

# one hot encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# combine input data with encoded geography
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)


prediction = model.predict(input_data)

if prediction:
    st.write(f'The predicted salary of customer is {prediction}')