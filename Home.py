import streamlit as st
import pandas as pd

st.markdown(
    """# Why this app?
Interpreting predictions made by a Machine Learning model is sometimes difficult 
and confusing. **Misinterpretation** can lead to **wrong business decisions** with negative
consequences for the business (and for your career ;)).

# How?
This app is a visual answer to "how should I deal with my model results".

# What?
Considering a churn prediction problem
for a telecommunication company (=are customers going to leave?), we investigate:
* the model output (probabilities)
* the decision threshold concept 
* its consequence on model performance


# Input Data
This is what the input data looks like:
"""
)
st.dataframe(
    pd.read_csv("./data/predictions_logistic_regression.csv", index_col=0).head()
)
