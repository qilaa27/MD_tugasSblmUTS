import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# Function to load the trained model
def load_model(filename):
    try:
        model = joblib.load(filename)
        return model
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

# Function to make predictions using the model
def predict_with_model(model, user_input):
    try:
        prediction = model.predict([user_input])
        prediction_prob = model.predict_proba([user_input])
        return prediction[0], prediction_prob[0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None, None

def main():
    # App title and description
    st.title('Machine Learning App')
    st.info('This app will predict your obesity level!')

    # Load the dataset
    data = pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')  # Update path if necessary

    # Display raw data in an expandable section
    with st.expander("**Data**"):
        st.markdown("This is a raw data")
        st.dataframe(data)

    # Data visualization (Weight vs Height)
    with st.expander("**Data Visualization**"):
        plt.figure(figsize=(10, 6))
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

    # Input form for numeric and categorical data
    st.subheader("Data input by user")

    # Numeric inputs using st.slider
    age = st.slider("Age", min_value=18, max_value=100, value=25, step=1)
    height = st.slider("Height (m)", min_value=1.0, max_value=2.5, value=1.65, step=0.01)
    weight = st.slider("Weight (kg)", min_value=30, max_value=200, value=70, step=1)

    # Categorical input using st.selectbox
    gender = st.selectbox("Gender", ["Male", "Female"])
    family_history = st.selectbox("Family history with overweight", ["yes", "no"])
    favc = st.selectbox("FAVC (Frequency of high-calorie food consumption)", ["yes", "no"])
    fcvc = st.slider("FCVC (Frequency of consumption of vegetables)", min_value=1, max_value=5, value=3, step=1)
    ncp = st.slider("NCP (Number of meals per day)", min_value=1, max_value=6, value=3, step=1)
    caec = st.selectbox("CAEC (Work related physical activity)", ["Sometimes", "Frequently", "No"])

    # Prepare the input for prediction (convert categorical to numeric as needed)
    gender = 1 if gender == 'Male' else 0
    family_history = 1 if family_history == 'yes' else 0
    favc = 1 if favc == 'yes' else 0
    caec = {"Sometimes": 2, "Frequently": 1, "No": 0}[caec]

    # Create the user input data list
    user_input = [gender, age, height, weight, family_history, favc, fcvc, ncp, caec]

    # Display the user input data
    st.subheader("Data input by user")
    input_data = {
        "Gender": ["Male" if gender == 1 else "Female"],
        "Age": [age],
        "Height": [height],
        "Weight": [weight],
        "family_history_with_overweight": ["yes" if family_history == 1 else "no"],
        "FAVC": ["yes" if favc == 1 else "no"],
        "FCVC": [fcvc],
        "NCP": [ncp],
        "CAEC": ["Sometimes" if caec == 2 else "Frequently" if caec == 1 else "No"]
    }
    input_df = pd.DataFrame(input_data)
    st.write(input_df)

    # Load the trained model
    model = load_model('trained_model.pkl')

    # Button to predict when clicked
    if st.button('Predict'):
        if model is not None:
            # Get prediction and probabilities
            prediction, probabilities = predict_with_model(model, user_input)

            if prediction is not None:
                # Show classification probabilities
                st.subheader("Obesity Prediction")
                prob_df = pd.DataFrame([probabilities], columns=["Insufficient Weight", "Normal Weight", "Overweight Level I", "Overweight Level II", "Obesity Type I", "Obesity Type II", "Obesity Type III"])
                st.dataframe(prob_df)

                # Show final prediction
                st.write(f"The predicted output is: {prediction}")
        else:
            st.error("Model could not be loaded. Please check the model file.")

if __name__ == '__main__':
    main()
