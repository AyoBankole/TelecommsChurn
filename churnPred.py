import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import streamlit as st
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# import data
df= pd.read_csv('expresso_processed.csv')
data= df.copy()
# import the model
model= joblib.load('expressochurn.pkl')
# import encoder
encoder= joblib.load('encoder.pkl')
df.drop(['Unnamed: 0', 'MRG'], axis = 1, inplace = True)

for i in df.drop('CHURN', axis = 1).columns:
    if df[i].dtypes == 'O':
        df[i] = encoder.fit_transform(df[i])


x = df.drop('CHURN', axis = 1)
y = df.CHURN

xtrain, xtest, ytrain, ytest = train_test_split(x,y, train_size=0.8,test_size= 0.20, stratify= y)

model = DecisionTreeClassifier(random_state=1)
model.fit(xtrain, ytrain)

st.markdown("<h1 style = 'color: #12372A; text-align: center; font-family: Sans serif '>TELECOM CHURN</h1>", unsafe_allow_html = True)
st.markdown("<h4 style = 'margin: -30px; color: #561C24; text-align: center; font-family: Roboto '>Built By The Mushin Data Guy</h4>", unsafe_allow_html = True)
st.image('transparent-money-business-finance-graph-data-6575327c9c0d78.7872553217021794526392.png', width = 250, use_column_width = True )
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h4 style = 'color: #1F4172; text-align: center; font-family:cursive '>Project Overview</h4>", unsafe_allow_html = True)
st.markdown("<p style = 'text-align: justify'>The predictive telecommunications customer attrition modeling project aims to leverage machine learning techniques to develop an accurate and robust model capable of predicting the whether a customer attricts or not. By analyzing historical data, identifying key features influencing customer's descision, and employing advanced classification algorithms, the project seeks to provide valuable insights for business analysts, entrepreneur, large and small scale businesses. The primary objective of this project is to create a reliable machine learning model that accurately predicts customer's decision based on relevant features such as location, income in ($), client's duration, and other influencing factors. The model should be versatile enough to adapt to different business plans, providing meaningful predictions for a wide range of businesses.", unsafe_allow_html=True)
st.sidebar.image('pngwing.com.png' ,width = 150, use_column_width = True, caption= 'Welcome User')
st.markdown("<br>", unsafe_allow_html = True)

tenure = st.sidebar.selectbox('DURATION AS A CUSTOMER', data.TENURE.unique())
montant = st.sidebar.number_input('AMOUNT RELOADED', df.MONTANT.min(), df.MONTANT.max())
freq_rech = st.sidebar.number_input('RELOADS', df.FREQUENCE_RECH.min(), df.FREQUENCE_RECH.max())
revenue = st.sidebar.number_input('MONTHLY INCOME', df.REVENUE.min(), df.REVENUE.max())
arpu_segment = st.sidebar.number_input('INCOME(90 DAYS)', df.ARPU_SEGMENT.min(), df.ARPU_SEGMENT.max())
frequence = st.sidebar.number_input('INCOME FREQUENCY', df.FREQUENCE.min(), df.FREQUENCE.max())
data_volume = st.sidebar.number_input('ACTIVENESS OF CLIENT(90 DAYS)', df.DATA_VOLUME.min(), df.DATA_VOLUME.max())
no_net = st.sidebar.number_input('CALL DURATION', df.ON_NET.min(), df.ON_NET.max())
regularity = st.sidebar.number_input('REGULARITY', df.REGULARITY.min(), df.REGULARITY.max())

new_tenure = encoder.transform([tenure])

input_var = pd.DataFrame({'TENURE': [new_tenure],
                           'MONTANT': [montant], 
                           'FREQUENCE_RECH': [freq_rech],
                          'REVENUE':[revenue],
                           'ARPU_SEGMENT':[arpu_segment],
                            'FREQUENCE':[frequence],
                             'DATA_VOLUME':[data_volume],
                              'ON_NET':[no_net],
                                'REGULARITY':[regularity]})
st.markdown("<br>", unsafe_allow_html= True)
st.markdown("<h5 style= 'margin: -30px; color:olive; font:sans serif' >", unsafe_allow_html= True)
st.dataframe(input_var)

predicted = model.predict(input_var)
output = None
if predicted[0] == 0:
    output = 'Not Churn'
else:
    output = 'Churn'
# transformed= encoder.transform([predicted])
prediction, interprete = st.tabs(["Model Prediction", "Model Interpretation"])
with prediction:
    pred = st.button('Push To Predict')
    if pred: 
        st.success(f'The customer is predicted to {output}')
        st.balloons()