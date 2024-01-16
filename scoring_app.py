import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from utils import LogOddsEncoder

cols = ['duration',
 'loan_amt',
 'installment_rate',
 'age',
 'checking_acc_status',
 'cred_hist',
 'saving_acc_bonds',
 'present_employment_since',
 'personal_stat_gender',
 'other_debtors_guarantors',
 'other_installment_plans',
 'is_foreign_worker',
 'purpose']


# Load the pickled pipeline
pipeline = load('pipeline.pkl')

data = pd.read_csv('statlog+german+credit+data/german.data', sep=' ', header=None)
columns = ['checking_acc_status', 'duration', 'cred_hist', 'purpose', 'loan_amt', 'saving_acc_bonds',
          'present_employment_since','installment_rate', 'personal_stat_gender', 'other_debtors_guarantors',
          'present_residence_since', 'property', 'age', 'other_installment_plans', 'housing', 'num_curr_loans',
          'job', 'num_people_provide_maint', 'telephone', 'is_foreign_worker', 'target']
df = pd.DataFrame(data.values, columns=columns)

mapping_dict = {
    'checking_acc_status': {'A11': 'below_0', 'A12': 'below_200', 'A13': 'above:200', 'A14': 'no_cheking_acc'},
    'cred_hist': {'A30': 'no_loan_or_paid_duly_other', 'A31': 'paid_duly_this_bank', 'A32': 'curr_loans_paid_duly',
                  'A33': 'delay_in_past', 'A34': 'risky_acc_or_curr_loan_other'},
    'purpose': {'A40': 'car_new', 'A41': 'car_used', 'A42': 'furniture_equipment', 'A43': 'radio_tv',
                'A44': 'domestic_applience', 'A45': 'repairs', 'A46': 'education', 'A47': 'vacation',
                'A48': 'retraining', 'A49': 'business', 'A410': 'others'},
    'saving_acc_bonds': {'A61': 'below_100', 'A62': 'below_500', 'A63': 'below_1000', 'A64': 'above_1000',
                         'A65': 'unknown_no_saving_acc'},
    'present_employment_since': {'A71': 'unemployed', 'A72': 'below_1y', 'A73': 'below_4y', 'A74': 'below_7y', 'A75': 'above_7y'},
    'personal_stat_gender': {'A91': 'male:divorced', 'A92': 'female:divorced_or_married', 'A93': 'male:single',
                      'A94': 'male:married_or_widowed', 'A95': 'female:single'},
    'other_debtors_guarantors': {'A101': 'none', 'A102': 'co_applicant', 'A103': 'guarantor'},
    'property': {'A121': 'real_estate', 'A122': 'life_insurance_or_aggreements', 'A123': 'car_or_other',
                 'A124': 'unknown_or_no_property'},
    'other_installment_plans': {'A141': 'bank', 'A142': 'store', 'A143': 'none'},
    'housing': {'A151': 'rent', 'A152': 'own', 'A153': 'for_free'},
    'job': {'A171': 'unemployed_non_resident', 'A172': 'unskilled_resident', 'A173': 'skilled_official',
            'A174': 'management_or_self_emp'},
    'telephone': {'A191': 'no', 'A192': 'yes'},
    'is_foreign_worker': {'A201': 'yes', 'A202': 'no'},
    'target':{1:'good',2:'bad'}
}

df.replace(mapping_dict, inplace=True)
numeric_features = ['duration', 'loan_amt', 'installment_rate', 'present_residence_since', 'age', 'num_curr_loans', 
                    'num_people_provide_maint']
categorical_features = ['checking_acc_status', 'cred_hist', 'purpose', 'saving_acc_bonds', 'present_employment_since', 
                        'personal_stat_gender', 'other_debtors_guarantors', 'property', 'other_installment_plans', 
                        'housing', 'job', 'telephone', 'is_foreign_worker']

df[numeric_features] = df[numeric_features].apply(pd.to_numeric, errors='coerce', downcast='integer')
df[categorical_features] = df[categorical_features].astype('category')

# Streamlit App
st.title('Credit Score Prediction App')               
        
# Sidebar
st.sidebar.title('User Input')
categorical_input = {}
for feature in categorical_features:
    categories = df[feature].unique()
    selected_category = st.sidebar.selectbox(f'Select {feature}', categories)
    categorical_input[feature] = selected_category


numeric_input = {}
for feature in numeric_features:
    value = st.sidebar.number_input(f'Enter {feature}', value=0)
    numeric_input[feature] = value


if st.sidebar.button('Calculate Credit Score'):
    user_data = pd.DataFrame({**numeric_input, **categorical_input}, index=[0])
    proba = pipeline.predict_proba(user_data)
    
     # Main part
    st.title('Credit Score Prediction Result')
    st.success(f'Predicted Credit Score: {np.round(proba[0][1], 4)}')
    
    ud2 = pd.DataFrame(user_data[cols].T.values, columns=['raw_data'], index=cols)
    transformed1 = pipeline[:-1].transform(user_data)
    
    selected = pipeline[-1].feature_names_in_
    try:
        ud2['transformed'] = transformed1.values.T
    except:
        st.write(transformed1)
    ud2['coefficient'] = pipeline[-1].coef_[0]
    ud2['contribution'] = ud2['transformed'] * ud2['coefficient']
    
    
    
    ud_intercept = pd.DataFrame(np.array([1, 1, pipeline[-1].intercept_[0]]),
                                columns=['intercept'], index=['raw_data','transformed','coefficient'])
    
    ud = pd.concat([ud2, ud_intercept.T], axis = 0)
    
    st.write(1/(1+np.exp(-ud['contribution'].sum())))
    st.dataframe(ud)
    
   
