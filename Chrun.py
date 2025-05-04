import pickle
import streamlit as st

model = pickle.load(open('churn.sav', 'rb'))

st.title('Customer Churn Modeling')

CreditScore = st.number_input('CreditScore')
Geography = st.number_input('Geography')
Gender = st.number_input('BloodPressure')
Age = st.number_input('Age')
Tenure = st.number_input('Tenure')
Balance = st.number_input('Balance')
NumOfProducts = st.number_input('NumOfProducts')
HasCrCard = st.number_input('HasCrCard')
IsActiveMember = st.number_input('IsActiveMember')
EstimatedSalary = st.number_input('EstimatedSalary')

predict = ''

if st.button('Customer Churn Modeling'):
    predict = model.predict(
        [[CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary]]
    )
    st.write('Customer Churn Modeling: ', predict)