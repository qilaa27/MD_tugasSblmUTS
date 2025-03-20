import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_model(filename):
    model = joblib.load(filename)
    return model

def predict_with_model(model, user_input):
    prediction = model.predict([user_input])
    return prediction[0]

def main():
    st.title('Machine Learning App')
    st.info('This app will predict your obesity level!')

    data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

    with st.expander("# **Data**"):
        st.markdown("This is a raw data") 
        st.dataframe(data)

    with st.expander("# **Data Visualization**"):
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=data, x='Height', y='Weight', hue='NObeyesdad', palette='Set1', s=100, alpha=0.7)
        plt.title('Weight vs Height by Obesity Class')
        plt.xlabel('Height (m)')
        plt.ylabel('Weight (kg)')

        st.pyplot(plt)

if __name__ == '__main__':
    main()
