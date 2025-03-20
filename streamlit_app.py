import streamlit as st
import pandas as pd
import numpy as np
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
    
        # Define color mapping for each category in 'NObeyesdad'
        colors = {'Insufficient_Weight': 'blue', 'Normal_Weight': 'green', 'Obesity_Type_I': 'red', 'Obesity_Type_II': 'orange', 'Obesity_Type_III': 'purple'}

        # Plot each category with the corresponding color
        for category, color in colors.items():
            subset = data[data['NObeyesdad'] == category]
            plt.scatter(subset['Height'], subset['Weight'], label=category, color=color, s=100, alpha=0.7)

        # Adding labels and title
        plt.title('Weight vs Height by Obesity Class')
        plt.xlabel('Height (m)')
        plt.ylabel('Weight (kg)')

        # Show the legend
        plt.legend(title='Obesity Class')

        # Display the plot
        st.pyplot(plt)

if __name__ == '__main__':
    main()
