import streamlit as st 
import joblib
import numpy as np 
import pandas as pa 
import shap
import pandas as pd
import matplotlib.pyplot as plt
# Load the new model
model = joblib.load ('XGBoost.pkl')
# Load the test data from X_test.csv to create LIME explainer
X_test = pd.read_csv('clinic.csv')
# Define feature names from the new dataset
feature_names =['Age',	'TStage',	'NStage',	'PRStatus',	'HER2Status',	'Ki67Index',	'HistologicalGrading',	'LengthofMassmm',	'Margin',	'SpiculeSign',	'TICPattern',	'MRIReportedALNM',	'RBC',	'WBC',	'Monocytes',	'CA153',	'CEA']	


# Streamlit user interface
st.title("Breast cancer metastasis")
# Age: numerical input
# Sex: categorical selection
# Chest Pain Type (ср): categorical selection
Age = st.number_input("Age:", min_value=0.0, max_value=200.0, value=50.0)
TStage = st.selectbox("TStage:",options=[0, 1, 2, 3,4])
NStage = st.selectbox("NStage:",options=[0, 1, 2, 3,4])
PRStatus = st.selectbox("PRStatus:",options=[0, 1])
HER2Status = st.selectbox("HER2Status:",options=[0, 1,2])
Ki67Index = st.number_input("Ki67Index:", min_value=0.0, max_value=1000.0, value=50.0)
HistologicalGrading = st.selectbox("HistologicalGrading:",options=[0, 1,2,3,4])
LengthofMassmm = st.number_input("LengthofMassmm:", min_value=0.0, max_value=1000.0, value=50.0)
Margin = st.selectbox("Margin:",options=[0, 1,2,3,4])
SpiculeSign = st.selectbox("SpiculeSign:",options=[0, 1])
TICPattern = st.selectbox("TICPattern:",options=[0, 1,2,3])
MRIReportedALNM = st.selectbox("MRIReportedALNM:",options=[0, 1,2])
RBC_value = st.number_input("RBC:", min_value=0.0, max_value=1000.0, value=50.0)
WBC_value = st.number_input("WBC:", min_value=0.0, max_value=1000.0, value=50.0)
Monocytes_value = st.number_input("Monocytes:", min_value=0.0, max_value=1000.0, value=5.0)
CA153_value = st.number_input("CA153:", min_value=0.0, max_value=1000.0, value=50.0)
CEA_value = st.number_input("CEA:", min_value=0.0, max_value=1000.0, value=50.0)



feature_values = [Age ,
TStage ,
NStage,
PRStatus,
HER2Status ,
Ki67Index ,
HistologicalGrading ,
LengthofMassmm ,
Margin,
SpiculeSign ,
TICPattern,
MRIReportedALNM ,
RBC_value ,
WBC_value ,
Monocytes_value ,
CA153_value ,
CEA_value]
features=np.array(feature_values)
predicted_class=1
if st.button("Predict"):  
    # 将 feature 数组从一维转为二维  
    features = features.reshape(1, -1)  

    # 预测类别和分类概率  
    predicted_class = model.predict(features)[0]  
    predicted_proba = model.predict_proba(features)[0]  

    # 根据预测结果生成建议  
    probability = predicted_proba[predicted_class] * 100  

    if predicted_class == 1:  
        advice = (f"According to the model, your risk of tumor metastasis is high")  
    else:  
        advice = (f"According to the model, your risk of tumor metastasis is low")  
    
    st.write(advice)  
# SHAP Explanation

