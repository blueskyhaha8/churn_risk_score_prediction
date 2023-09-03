import streamlit as st
import numpy as np
import pickle

# Nilai-nilai yang mungkin ada dalam setiap kolom
gender_options = ['F', 'M']
region_category_options = ['Village', 'City', 'Town']
membership_category_options = ['Platinum Membership', 'Premium Membership', 'No Membership', 'Gold Membership', 'Silver Membership', 'Basic Membership']
joined_through_referral_options = ['No', 'Yes']
used_special_discount_options = ['Yes', 'No']
offer_application_preference_options = ['Yes', 'No']
past_complaint_options = ['No', 'Yes']
complaint_status_options = ['Not Applicable', 'Solved', 'Solved in Follow-up', 'Unsolved', 'No Information Available']
feedback_options = ['Products always in Stock', 'Quality Customer Care', 'Poor Website', 'No reason specified', 'Poor Product Quality', 'Poor Customer Service', 'Too many ads', 'User Friendly Website', 'Reasonable Price']

# Load your pre-trained models and scalers
with open('model/scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('model/log_transformed_ATV.pkl', 'rb') as log_file:
    log_transformed_ATV = pickle.load(log_file)

with open('model/log_transformed_PIW.pkl', 'rb') as log_file:
    log_transformed_PIW = pickle.load(log_file)

with open('model/log_transformed_ATS.pkl', 'rb') as log_file:
    log_transformed_ATS = pickle.load(log_file)

with open('model/random_forest_model.pkl', 'rb') as model_file:
    rf_model_2a = pickle.load(model_file)

# Streamlit UI
st.title("Churn Risk Score Prediction for HackelEarth Website")

# Create input fields for user data
customer_id = st.text_input('Customer ID', '')
age = st.slider('Age', 10, 100, 25)
gender = st.selectbox('Gender', gender_options)
region_category = st.selectbox('Region Category', region_category_options)
membership_category = st.selectbox('Membership Category', membership_category_options)
joined_through_referral = st.selectbox('Joined Through Referral', joined_through_referral_options)
avg_time_spent = st.number_input('Average Time Spent', min_value=0)
avg_transaction_value = st.number_input('Average Transaction Value', min_value=0)
avg_frequency_login_days = st.number_input('Average Frequency Login Days', min_value=0)
points_in_wallet = st.number_input('Points in Wallet', min_value=0)
used_special_discount = st.selectbox('Used Special Discount', used_special_discount_options)
offer_application_preference = st.selectbox('Offer Application Preference', offer_application_preference_options)
past_complaint = st.selectbox('Past Complaint', past_complaint_options)
complaint_status = st.selectbox('Complaint Status', complaint_status_options)
feedback = st.selectbox('Feedback', feedback_options)

# Make prediction when the user clicks the "Submit" button
if st.button('Submit'):
    # Prepare input data

    # Apply LabelEncoder to membership_category
    MC =  0 if membership_category== 'Basic Membership' else (1 if membership_category=='Gold Membership' 
                                                               else (2 if membership_category=='No Membership'
                                                                     else(3 if membership_category=='Platinum Membership'
                                                                          else(4 if membership_category=='Premium Membership'
                                                                               else 5 ))))

    # Apply LabelEncoder to gender
    GEN = 0 if gender== 'F' else 1
    
    # Apply LabelEncoder to region_category
    RC = 0 if region_category=='City'else (1 if region_category=='Town'else 2)
    
    # Apply LabelEncoder to joined_through_referral
    JTR = 0 if joined_through_referral=='No' else 1
    
    # Apply LabelEncoder to used_special_discount
    USD = 0 if used_special_discount=='No' else 1
    
    # Apply LabelEncoder to offer_application_preference
    OAP = 0 if offer_application_preference=='No' else 1

        # Apply LabelEncoder to past_complaint
    PC = 0 if past_complaint=='No' else 1

        # Apply LabelEncoder to complaint_status
    CS = 0 if complaint_status=='No Information Available'else(1 if complaint_status=='Not Applicable' 
                                                               else (2 if complaint_status=='Solved'
                                                                     else (3 if complaint_status=='Solved in Follow-up'
                                                                           else 4)))
    
    # Apply LabelEncoder to feedback
    fb = 0 if feedback=='No reason specified' else (1 if feedback=='Poor Customer Service'
                                                    else (2 if feedback=="Poor Product Quality"
                                                          else(3 if feedback=='Poor Website'
                                                               else(4 if feedback=="Products always in Stock"
                                                                    else (5 if feedback=='Quality Customer Care'
                                                                          else (6 if feedback=='Reasonable Price'
                                                                                else (7 if feedback=="Too many ads"
                                                                                      else 8)))))))
    
    
   
    # Combine input data
    input_data=[age, GEN, RC, MC,JTR, avg_frequency_login_days, USD, OAP, PC, CS, 
                fb,points_in_wallet, avg_time_spent, avg_transaction_value]
    
    scaled_input_data = scaler.transform(np.array([input_data]))


   
    # Perform prediction with the Random Forest model
    predicted_churn_risk_score = rf_model_2a.predict(scaled_input_data)[0]

    # Display the predicted Churn Risk Score to the user
    st.write('Predicted Churn Risk Score:', predicted_churn_risk_score)
