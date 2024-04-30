import os, tempfile
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib, pickle


# Streamlit app
st.subheader('Survival (within 2 years) Predictor After Having a Heart Attack')

uploaded_file = st.file_uploader("Upload your echocardiogram.test to get your predictions", label_visibility="collapsed")


if st.button("Submit"):

    # Validate inputs
    try:
        if uploaded_file is not None:
            # Save uploaded file temporarily to disk, load and split the file into pages, delete temp file
            with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                tmp_file.write(uploaded_file.read())

            # Read the data file into a DataFrame
            dataframe = pd.read_csv(tmp_file.name, header=None) # Assuming there's no header in the file
            os.remove(tmp_file.name)

            # Define column names
            column_names = ["survival", "still-alive", "age-at-heart-attack", "pericardial-effusion", "fractional-shortening",
                        "epss", "lvdd", "wall-motion-score", "wall-motion-index", "mult", "name", "group", "alive-at-1"] # Add all column names here

            # Assign column names to the DataFrame
            dataframe.columns = column_names

            # Drop irrelevant columns
            df = dataframe.drop(["survival", "still-alive", "name", "group", "mult", "wall-motion-score", "alive-at-1"], axis=1)

            # Convert object columns to float
            df = df.apply(pd.to_numeric, errors='coerce').copy()
            #df.apply(pd.to_numeric, errors='coerce')

            # Handling missing values
            df.dropna(inplace=True)

            # Define age groups
            bins = [0, 55, 65, 100]  # Define the age boundaries for each group
            labels = ['0-55', '55-65', '65-100']  # Define labels for each group

            # Create a new column for age groups
            df['age_group'] = pd.cut(df['age-at-heart-attack'], bins=bins, labels=labels, right=False)

            # Let's add wall-motion-index and lvdd interaction term
            df['wall-motion-index_lvdd_interaction'] = df['wall-motion-index'] * df['lvdd']

            # Let's add age at heart attack and lvdd interaction term
            df['age_lvdd_interaction'] = df['age-at-heart-attack'] * df['lvdd']

            # Change the order of DataFrame columns to prepare it for Label Encoding for the 'age_group' and 'pericardial-effusion' column
            df = df[[
                'age_group',
                'pericardial-effusion',
                'age-at-heart-attack',
                'fractional-shortening',
                'epss',
                'lvdd',
                'wall-motion-index',
                'age_lvdd_interaction',
                'wall-motion-index_lvdd_interaction']].copy()

            # Encoding categorical data -> age_group and pericardial-effusion
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0, 1])], remainder='passthrough') # [0, 1] refers to age_group and pericardial-effusion columns
            df_array = np.array(ct.fit_transform(df))
            dff = pd.DataFrame(df_array, columns=ct.get_feature_names_out(), index=df.index)

            # Feature Scaling
            scaler_filename = "model/standard_scaler.gz"
            sc = joblib.load(scaler_filename)
            dff.iloc[:, 5:] = sc.transform(dff.iloc[:, 5:].values) # Excluding the first two columns (categorical variables) and the last target column

            # load the models from disk
            filename = 'model/RandomForest.sav'
            rf_best_estimator = pickle.load(open(filename, 'rb'))

            filename = 'model/XGBoost.sav'
            xgboost_best_estimator = pickle.load(open(filename, 'rb'))

            # Predictions by Random Forest 
            y_pred_rf = rf_best_estimator.predict(dff)
            y_pred_prob_rf = np.max(rf_best_estimator.predict_proba(dff), axis=1)

            # Predictions by XGBoost Classifier
            y_pred_xg = xgboost_best_estimator.predict(dff)
            y_pred_prob_xg = np.max(xgboost_best_estimator.predict_proba(dff), axis=1)

            # Adding the predictions into the dataframe
            dff["Prediction RF"] = y_pred_rf.astype(int)
            dff["Prediction Probability RF"] = y_pred_prob_rf

            dff["Prediction XGBoost"] = y_pred_xg.astype(int)
            dff["Prediction Probability XGBoost"] = y_pred_prob_xg

            prediction_df = dff[['Prediction RF',
                'Prediction Probability RF',
                'Prediction XGBoost',
                'Prediction Probability XGBoost']]

            # concatenating df_labelled and df_unlabelled along rows
            test_data_df_updated = pd.concat([dataframe, prediction_df], axis=1)

            # Display the DataFrame
            st.write(test_data_df_updated)

        else:
            # Read the data file into a DataFrame
            df = pd.read_csv("data/echocardiogram_predictions.csv", header=None) # Assuming there's no header in the file

            # Display the DataFrame
            st.write(df)

    except Exception as e:
        st.error(f"An error occurred: {e}")

