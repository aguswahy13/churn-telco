import streamlit as st
import pickle
import csv
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load your training data
train_data = pd.read_csv('Data_Train_Churn.csv')

# Assume the last column is the target variable
X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)  # Set a random state for reproducibility
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Create a sample model
model = RandomForestClassifier(random_state=42)  # Set a random state for reproducibility

# Train the model
model.fit(X_train_resampled, y_train_resampled)

# Save the model to a pickle file
with open('model_rf_smote.pkl', 'wb') as file:
    pickle.dump(model, file, protocol=pickle.HIGHEST_PROTOCOL)


"""
# Load the trained model
with open('model_rf_smote.pkl', 'rb') as file:
    model = pickle.load(file, encoding='latin1')
"""

# Define a function to make predictions
def predict(account_length, international_plan, voice_mail_plan, number_vmail_messages,
            total_day_minutes, total_day_calls, total_eve_minutes, total_eve_calls,
            total_night_minutes, total_night_calls, total_intl_minutes, total_intl_calls,
            number_customer_service_calls):
    
    # Create a numpy array of inputs
    inputs = np.array([[account_length, international_plan, voice_mail_plan, number_vmail_messages,
                        total_day_minutes, total_day_calls, total_eve_minutes, total_eve_calls,
                        total_night_minutes, total_night_calls, total_intl_minutes, total_intl_calls,
                        number_customer_service_calls]])

    # Make a prediction using the model
    prediction = model.predict(inputs)
    
    return prediction[0]

# Streamlit app layout
st.title('Customer Churn Prediction')

st.header('Enter the customer details:')
account_length = st.number_input('Account Length', min_value=0, max_value=100, value=0)
international_plan = st.selectbox('International Plan', [0, 1])
voice_mail_plan = st.selectbox('Voice Mail Plan', [0, 1])
number_vmail_messages = st.number_input('Number of Voice Mail Messages', min_value=0, max_value=100, value=0)
total_day_minutes = st.number_input('Total Day Minutes', min_value=0.0, max_value=1000.0, value=0.0)
total_day_calls = st.number_input('Total Day Calls', min_value=0, max_value=1000, value=0)
total_eve_minutes = st.number_input('Total Evening Minutes', min_value=0.0, max_value=1000.0, value=0.0)
total_eve_calls = st.number_input('Total Evening Calls', min_value=0, max_value=1000, value=0)
total_night_minutes = st.number_input('Total Night Minutes', min_value=0.0, max_value=1000.0, value=0.0)
total_night_calls = st.number_input('Total Night Calls', min_value=0, max_value=1000, value=0)
total_intl_minutes = st.number_input('Total International Minutes', min_value=0.0, max_value=100.0, value=0.0)
total_intl_calls = st.number_input('Total International Calls', min_value=0, max_value=100, value=0)
number_customer_service_calls = st.number_input('Number of Customer Service Calls', min_value=0, max_value=100, value=0)

if st.button('Predict Churn'):
    result = predict(account_length, international_plan, voice_mail_plan, number_vmail_messages,
                     total_day_minutes, total_day_calls, total_eve_minutes, total_eve_calls,
                     total_night_minutes, total_night_calls, total_intl_minutes, total_intl_calls,
                     number_customer_service_calls)
    if result == 1:
        st.write('The customer is likely to churn.')
    else:
        st.write('The customer is not likely to churn.')

# To run the Streamlit app, use the command:
# streamlit run <name