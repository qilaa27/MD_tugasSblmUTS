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
    data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    # Make "Data" bold using markdown
    st.markdown("### **Data**")  # Make the title "Data" bold
    
    # Create an expandable section
    with st.expander("Click to view the raw data >"):
        st.dataframe(data)  # Displaying the dataset in a dataframe

if __name__ == '__main__':
    main()
