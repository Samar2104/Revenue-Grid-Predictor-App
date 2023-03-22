# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 12:21:14 2023

@author: Asus
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest

# load dataset
finance_database= pd.read_csv("C:/Users/Asus/Desktop/DataSet/finance_train.csv")
finance_database.head()
#removing of columns with unique elements and using correlation analysis
finance_database.drop(['REF_NO','year_last_moved','post_code','post_area' ],axis=1,inplace=True)
finance_database['children'] = finance_database['children'].str.replace('Zero', '0').str.replace('+','')
finance_database['children'] = finance_database['children'].astype(float)
finance_database['age_band']=finance_database['age_band'].str.replace('+','-75')
#finance_database['age_band_min','age_band_max']=finance_database['age_band'].str.split('-', expand=True)

finance_database['age_band']=finance_database['age_band'].str.replace('Unknown','0-0')
finance_database['age_band_min']=finance_database['age_band'].str.split('-', expand=True)[0].astype(float)
finance_database['age_band_max']=finance_database['age_band'].str.split('-', expand=True)[1].astype(float)
finance_database['age_band_mean'] = (finance_database['age_band_min'] + finance_database['age_band_max']) / 2
finance_database.drop(['age_band','age_band_min','age_band_max' ],axis=1,inplace=True)

def data_encode(col):
    categories=finance_database[col].unique()
    cat=categories.tolist()
    a=[]
    for i in range(len(cat)):
        a.append(i)
    cat_dict = dict(zip(cat, a))
    finance_database[col].replace(cat_dict , inplace=True)
    return finance_database.head(20)
for i in ['status','occupation','occupation_partner','home_status','self_employed','self_employed_partner','TVarea','gender','region']:
    a=data_encode(i)
print(a)
finance_database['family_income']=finance_database['family_income'].str.replace('<','').str.replace(',','').str.replace(' ','')




#seperate the '-' and pull the minimum and maximum values
finance_database['family_income_min']=finance_database['family_income'].str.split('>=', expand=True)[0]
finance_database['family_income_max']=finance_database['family_income'].str.split('>=', expand=True)[1]
finance_database['family_income_min'].value_counts()
finance_database['family_income_max'].value_counts()
finance_database.head()
finance_database.info()

#this code will convert the string value into null value and change the datatype into float or numeric
finance_database.family_income_min=pd.to_numeric(finance_database.family_income_min,errors='coerce')
finance_database.family_income_max=pd.to_numeric(finance_database.family_income_max,errors='coerce')

finance_database.isnull().sum()

tempMedian=finance_database.loc[:,"family_income_min"].median()
finance_database["family_income_min"].fillna(tempMedian,inplace=True)

tempMedian=finance_database.loc[:,"family_income_max"].median()
finance_database["family_income_max"].fillna(tempMedian,inplace=True)

finance_database.isnull().sum()



#Find the  mean values of each row 'family_income'

finance_database['family_income_mean'] = (finance_database['family_income_min'] + finance_database['family_income_max']) / 2
finance_database.drop(['family_income','family_income_min','family_income_max'],axis=1,inplace=True)
finance_database['family_income_mean'].value_counts()

finance_database.info()
X=finance_database.drop('Revenue.Grid', axis=1)
y=finance_database['Revenue.Grid']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
from sklearn.metrics import accuracy_score #0.97
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: ", accuracy)
from sklearn.metrics import classification_report

# Classification Model Validation
Confusion_Mat = pd.crosstab(y_test, y_pred)
print(Confusion_Mat) # R, C format (Actual = testY, Predicted = Test_Pred)

# Validation on Testset
print(classification_report(y_test, y_pred)) # Actual, Predicted



import streamlit as st

import pandas as pd


custom_css = """
    <style>
        body {
            background-color: #f0f0f0;
        }
        h1 {
            color: #ff6600;
        }
    </style>
"""
st.write(f'<head>{custom_css}</head>', unsafe_allow_html=True)
st.write('''
<h1> Revenue Grid Prediction App</h1>

This app predicts the **Revenue Grid** of a customer 


''', unsafe_allow_html=True)
data={'children':0,
        'status':'Partner',
        'occupation':'Professional',
        'Occupation Partner':'Professional',
        'Home Status':'Rent Privately',
        'Self Employed':'Yes',
        'Self Employed Partner':'Yes',
        'TV area':'HTV',
        'Average Credit card Transaction': 5,
        'Balance Transfer':5,
        'Term Deposit':1,
        'Life Insurance':2,
        'Medical Insurance':2,
        'Average Account Balance':5,
        'Personal Loan':5,
        'Investment in Mutual Fund ':5,
        'Investment in Tax Saving Bond':25,
        'Home Loan':5,
        'Online Purchase Amount':2,
        'Gender':'Male',
        'Region':'South West',
        'Investment in Equity':8,
        'Investment in Commudity':6,
        'Investment in Derivative':4,
        'Portfolio Balance':8,
        'Age Band Mean':2,
        'Family Income ':1}
df=pd.DataFrame(columns=['children','status','occupation','Occupation Partner','Home Status','Self Employed','Self Employed Partner','TV area','Average Credit card Transaction','Balance Transfer','Term Deposit','Life Insurance','Medical Insurance','Average Account Balance','Personal Loan','Investment in Mutual Fund ','Investment in Tax Saving Bond','Home Loan','Online Purchase Amount','Gender','Region','Investment in Commudity','Investment in Equity','Investment in Derivative','Portfolio Balance','Age Band Mean','Family Income '])
st.sidebar.header('User Input Features')
df=pd.DataFrame(columns=['children','status','occupation','Occupation Partner','Home Status','Self Employed','Self Employed Partner','TV area','Average Credit card Transaction','Balance Transfer','Term Deposit','Life Insurance','Medical Insurance','Average Account Balance','Personal Loan','Investment in Mutual Fund ','Investment in Tax Saving Bond','Home Loan','Online Purchase Amount','Gender','Region','Investment in Commudity','Investment in Equity','Investment in Derivative','Portfolio Balance','Age Band Mean','Family Income '])
children=st.sidebar.selectbox('Children',(0,1,2,3,4))
status=st.sidebar.selectbox('Status',('Partner', 'Single/Never Married', 'Divorced/Separated', 'Widowed', 'Unknown'))
occupation=st.sidebar.selectbox('Occupation',('Professional', 'Retired', 'Manual Worker', 'Secretarial/Admin', 'Other', 'Business Manager', 'Housewife', 'Unknown', 'Student'))
occupation_partner=st.sidebar.selectbox('Occupation Partner',('Professional', 'Retired', 'Manual Worker', 'Secretarial/Admin', 'Other', 'Business Manager', 'Housewife', 'Unknown', 'Student'))
home_status=st.sidebar.selectbox('Home Status', ('Rent Privately', 'Own Home', 'Rent from Council/HA', 'Live in Parental Hom', 'Unclassified'))
self_employed=st.sidebar.selectbox('Self Employed',('Yes', 'No'))
self_employed_partner=st.sidebar.selectbox('Self Employed Partner',('Yes', 'No'))
TVarea=st.sidebar.selectbox('TV area' , ('HTV', 'Unknown', 'Anglia', 'Granada', 'TV South West', 'Tyne Tees', 'Grampian', 'Yorkshire', 'Central', 'Meridian', 'Carlton', 'Ulster', 'Scottish TV', 'Border'))
Average_Credit_Card_Transaction=st.sidebar.slider('Average Credit card Transaction',0,278,17)
Balance_Transfer=st.sidebar.slider('Balance Transfer',0,508,37)
Term_Deposit=st.sidebar.slider('Term Deposit',0,515,21)
Life_Insurance=st.sidebar.slider('Life Insurance',0,407,49)
Medical_Insurance=st.sidebar.slider('Medical Insurance',0,307,14)
Average_account_Balance=st.sidebar.slider('Average Account Balance',0,277,24)
Personal_Loan=st.sidebar.slider('Personal Loan',0,546,18)
Investment_in_Mutual_Fund=st.sidebar.slider('Investment in Mutual Fund ',0,427,33)
Investment_Tax_Saving_Bond=st.sidebar.slider('Investment in Tax Saving Bond',0,75,4)
Home_Loan=st.sidebar.slider('Home Loan',0,121,3)
Online_Purchase_Amount=st.sidebar.slider('Online Purchase Amount',0,479,8)
gender=st.sidebar.selectbox('Gender' ,('Male','Female','Unknown'))
region=st.sidebar.selectbox('Region' ,('South West', 'Unknown', 'East Anglia', 'North West', 'North', 'Scotland', 'West Midlands', 'East Midlands', 'South East', 'Northern Ireland', 'Wales', 'Isle of Man', 'Channel Islands'))
Investment_in_Commudity=st.sidebar.slider('Investment in Commudity',0,158,28)
Investment_in_Equity=st.sidebar.slider('Investment in Equity',0.0,107.92,3.45)
Investment_in_Derivative=st.sidebar.slider('Investment in Derivative',0.0,190.92,3.45)
Portfolio_Balance=st.sidebar.slider('Portfolio Balance',0.0,350.92,3.45)
age_band_mean=st.sidebar.slider('Age Band Mean',0,73,20)
family_income_mean=st.sidebar.slider('Family Income ',6000,35000,15000)
button_clicked = st.sidebar.button("Submit", key="my_button", help="Click to do something")
if button_clicked:      
    data={
        'children':children,
        'status':status,
        'occupation':occupation,
        'Occupation Partner':occupation_partner,
        'Home Status':home_status,
        'Self Employed':self_employed,
        'Self Employed Partner':self_employed_partner,
        'TV area':TVarea,
        'Average Credit card Transaction': Average_Credit_Card_Transaction,
        'Balance Transfer':Balance_Transfer,
        'Term Deposit':Term_Deposit,
        'Life Insurance':Life_Insurance,
        'Medical Insurance':Medical_Insurance,
        'Average Account Balance':Average_account_Balance,
        'Personal Loan':Personal_Loan,
        'Investment in Mutual Fund ':Investment_in_Mutual_Fund,
        'Investment in Tax Saving Bond':Investment_Tax_Saving_Bond,
        'Home Loan':Home_Loan,
        'Online Purchase Amount':Online_Purchase_Amount,
        'Gender':gender,
        'Region':region,
        'Investment in Equity':Investment_in_Equity,
        'Investment in Commudity':Investment_in_Commudity,
        'Investment in Derivative':Investment_in_Derivative,
        'Portfolio Balance':Portfolio_Balance,
        'Age Band Mean':age_band_mean,
        'Family Income ':family_income_mean
        }
df =df.append(data, ignore_index=True)
st.write(df)
df['Gender'].replace({'Female': 0, 'Male': 1, 'Unknown': 2}, inplace=True)

df['status'].replace({'Partner': 0, 'Single/Never Married': 1, 'Divorced/Separated': 2, 'Widowed': 3, 'Unknown': 4}, inplace=True)
df['occupation'].replace({'Professional': 0, 'Retired': 1, 'Manual Worker': 2, 'Secretarial/Admin': 3, 'Other': 4, 'Business Manager': 5, 'Housewife': 6, 'Unknown': 7, 'Student': 8}, inplace=True)
df['Occupation Partner'].replace({'Professional': 0, 'Retired': 1, 'Manual Worker': 2, 'Unknown': 3, 'Business Manager': 4, 'Secretarial/Admin': 5, 'Housewife': 6, 'Other': 7, 'Student': 8}, inplace=True)
df['Home Status'].replace({'Rent Privately': 0, 'Own Home': 1, 'Rent from Council/HA': 2, 'Live in Parental Hom': 3, 'Unclassified': 4}, inplace=True)
df['Self Employed'].replace({'Yes': 0, 'No': 1}, inplace=True)
df['Self Employed Partner'].replace({'Yes': 0, 'No': 1}, inplace=True)
df['TV area'].replace({'HTV': 0, 'Unknown': 1, 'Anglia': 2, 'Granada': 3, 'TV South West': 4, 'Tyne Tees': 5, 'Grampian': 6, 'Yorkshire': 7, 'Central': 8, 'Meridian': 9, 'Carlton': 10, 'Ulster': 11, 'Scottish TV': 12, 'Border': 13}, inplace=True)
df['Region'].replace({'South West': 0, 'Unknown': 1, 'East Anglia': 2, 'North West': 3, 'North': 4, 'Scotland': 5, 'West Midlands': 6, 'East Midlands': 7, 'South East': 8, 'Northern Ireland': 9, 'Wales': 10, 'Isle of Man': 11, 'Channel Islands': 12}, inplace=True)




pred=model.predict(df)


st.write('**Prediction**')




s=pred.tolist()
a=['Revenue Grid']
d = dict(zip(a,s))
d_pred=pd.Series(d)
final=d_pred.replace({1:'Good Revenue', 2:'Bad Revenue'})
st.write(final)

pred_b=model.predict_proba(df)
d_prob=pd.DataFrame(pred_b)
d_prob.rename(columns={0:'Good Revenue',1:'Bad Revenue'
                     }, inplace=True)
st.write('**Prediction Probability**')
st.write(d_prob)