# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:17:02 2022

@author: Rohith Challam
"""



import pandas as pd
import numpy as np
import streamlit as st
from pickle import load

st.title('Model prediction on global development measurement')

st.sidebar.header('User Input Parameters')

def user_input_features():
    Birth_Rate = st.sidebar.number_input("Insert Birth Rate")
    Business_Tax_Rate = st.sidebar.number_input("insert Business Tax Rate",max_value=60.00)
    CO2_Emissions = st.sidebar.number_input('enter CO2 Emissions value',min_value=1.00)
    Days_to_Start_Business = st.sidebar.number_input('enter no of Days to Start Business',max_value=80.00)
    Ease_of_Business = st.sidebar.number_input('enter Ease of Business value')
    Energy_Usage = st.sidebar.number_input('enter Energy Usage value',min_value=1.00)
    GDP = st.sidebar.number_input('Enter GDP',min_value=1000)
    Health_Exp_GDP = st.sidebar.number_input('Enter Health Exp % GDP value')
    Health_Exp_Capita = st.sidebar.number_input('Enter Health Exp/Capita',min_value=1.00)
    Hours_to_do_Tax = st.sidebar.number_input('enter Hours to do Tax',max_value=600.00)
    Infant_Mortality_Rate = st.sidebar.number_input('enter Infant Mortality Rate')
    Internet_Usage = st.sidebar.number_input('enter Internet Usage duration')
    Lending_Interest = st.sidebar.number_input('enter Lending Interest rate')
    Life_Expectancy_Female = st.sidebar.number_input('enter Life Expectancy Female value')
    Life_Expectancy_Male = st.sidebar.number_input('enter Life Expectancy Male value')
    Mobile_Phone_Usage = st.sidebar.number_input('enter Mobile Phone Usage duration')
    Population_Total = st.sidebar.number_input('enter Population Total',min_value=1000)
    Population_Urban = st.sidebar.number_input('enter Population Urban value')
    Tourism_Inbound = st.sidebar.number_input('enter Tourism Inbound value',min_value=1000)
    Tourism_Outbound = st.sidebar.number_input('enter Tourism Outbound value',min_value=1000)

    data = {'Birth Rate':Birth_Rate, 'Business Tax Rate':Business_Tax_Rate, 'log10_CO2 Emissions':np.log10(CO2_Emissions),
       'Days to Start Business':Days_to_Start_Business, 'Ease of Business':Ease_of_Business, 'log10_Energy Usage':np.log10(Energy_Usage),
       'log10_GDP':np.log10(GDP), 'Health Exp % GDP':Health_Exp_GDP,'log10_Health Exp/Capita': np.log10(Health_Exp_Capita),
       'Hours to do Tax':Hours_to_do_Tax, 'Infant Mortality Rate':Infant_Mortality_Rate, 'Internet Usage':Internet_Usage,
       'Lending Interest': Lending_Interest, 'Life Expectancy Female':Life_Expectancy_Female, 'Life Expectancy Male':Life_Expectancy_Male,
       'Mobile Phone Usage':Mobile_Phone_Usage, 'log10_Population Total':np.log10(Population_Total), 'Population Urban':Population_Urban,
       'log10_Tourism Inbound':np.log10(Tourism_Inbound),'log10_Tourism Outbound': np.log10(Tourism_Outbound)}
    features = pd.DataFrame(data,index = [0])
    return features 
df = user_input_features()
st.subheader('User Input parameters')
st.write(df) 

# load the model from disk
loaded_model = load(open('Random_forest.sav', 'rb'))
prediction = loaded_model.predict(df)
st.subheader('Predicted Result')


def analysis(prediction):
    if prediction == 0:
        return 'developed country'
    elif prediction == 1:
        return 'under-developed country'
    elif prediction == 2:
        return 'the other small country'
    else:
        return 'developing country'


st.write(analysis(prediction))





       