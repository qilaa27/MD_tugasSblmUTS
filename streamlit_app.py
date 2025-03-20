import streamlit as st
import pandas as pd
import joblib

def load_model(filename):
    model = joblib.load(filename)
    return model

def predict_with_model(model, user_input):
    prediction = model.predict([user_input])
    return prediction[0]

def main():
    st.title('Machine Learning App')
    st.info('This app will predict your obesity level!')

    # Load and display raw data
    data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')  # Update path if necessary
    st.write("Raw Data")
    st.dataframe(data)  # Displaying the dataset in a dataframe

    # Example: Display the first few rows of the dataset
    st.write("First 5 Rows of Data:")
    st.write(data.head())

    # Load model
    model = load_model('trained_model.pkl')  # Ensure the model is in the right location

    # User input
    age = st.slider('Age', 18, 100, 25)
    gender = st.selectbox('Gender', ['Male', 'Female'])
    # Add other features as required...

    # Prepare user input for prediction (this part will depend on the model's expected input format)
    user_input = pd.DataFrame({'Age': [age], 'Gender': [gender]})  # Add other features to the input

    # Predict using the model
    if st.button('Predict'):
        prediction = predict_with_model(model, user_input)
        st.write(f"Predicted obesity level: {prediction}")

if __name__ == '__main__':
    main()
