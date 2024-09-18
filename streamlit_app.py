import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Title of the web app
st.title('Abalone Age Prediction App')

# File uploader to load the dataset
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the dataset
    st.write("Dataset Preview:")
    st.dataframe(df.head())

    # Check if 'Rings' column is present
    if 'Rings' in df.columns:
        # Create 'Age' column based on 'Rings'
        df['Age'] = df['Rings'] + 1.5
        
        # Display basic statistics
        st.write("Basic Statistics:")
        st.write(df.describe())
        
        # Define features (drop Rings and Age) and target (Rings)
        X = df.drop(columns=['Rings', 'Age'])  # Use only the physical measurements
        y = df['Rings']  # Predict the number of rings
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create the regression model (Random Forest in this case)
        model = RandomForestRegressor()
        
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict the number of rings on the test set
        y_pred = model.predict(X_test)
        
        # Calculate the predicted age
        predicted_age = y_pred + 1.5
        
        # Display model performance metrics
        st.write("Model Performance Metrics:")
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred)}")
        st.write(f"R-squared (R2 Score): {r2_score(y_test, y_pred)}")
        
        # Button to show predictions
        if st.button('Show Predictions'):
            result_df = pd.DataFrame({
                'Actual Rings': y_test,
                'Predicted Rings': y_pred,
                'Predicted Age': predicted_age
            }).reset_index(drop=True)
            
            st.write("Prediction Results:")
            st.dataframe(result_df)
    else:
        st.error("The dataset must contain the 'Rings' column for prediction.")
else:
    st.write("Please upload a dataset to proceed.")
