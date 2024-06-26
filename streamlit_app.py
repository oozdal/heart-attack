import os, tempfile
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib, pickle
import plotly
import plotly.io as pio
import streamlit.components.v1 as components
import altair as alt
pio.templates.default = "none"


def read_from_json(json_file):
    return plotly.io.read_json(json_file)


def map_number_to_class(number):
    class_mapping = {
        0: "won't survive",
        1: "survives",
    }
    return class_mapping.get(number, 'Unknown')


light = '''
<style>
    .stApp {
    background-color: white;
    }
</style>
'''

dark = '''
<style>
    .stApp {
    background-color: black;
    }
</style>
'''

# Template Configuration
st.markdown(dark, unsafe_allow_html=True)

# Streamlit app
st.subheader("Two-year Survival Predictor After Heart Attack")

checkbox_val = st.radio("Do you have an echocardiogram.test file to upload and get your predictions?", ("Yes", "No"), index=1)
if checkbox_val == "Yes":
    uploaded_file = st.file_uploader("Upload your echocardiogram.test to get your predictions.", label_visibility="visible")
else:
    st.write("Please answer the following questions and then submit to get your prediction!")

    # Age
    age = st.slider('At what age did you have a heart attack?', 
                    min_value=int(35), max_value=int(86), step=1, value=63)
    
    # Fractional Shortening
    frac_short = st.slider('What is your fractional shortening? (A measure of contracility around the heart.)',
                    min_value=0.01, max_value=0.61, value=0.216845)

    # epss
    epss = st.slider('What is your epss value? (E-point septal separation is another measure of contractility.)', 
                    min_value=0.0, max_value=40.0, value=12.446646)
    
    # lvdd
    lvdd = st.slider('What is your lvdd value? (Left ventricular end-diastolic dimension. This is a measure of the size of the heart at end-diastole.)', 
                    min_value=3.10, max_value=6.78, value=4.77)

    # wall motion index
    wall_motion_index = st.slider('What is your wall motion index? (Equals to wall-motion-score divided by number of segments seen.)',
                    min_value=1.00, max_value=3.00, value=1.398541)

    # pericardial effusion
    pericardial_effusion = st.radio("Do you have pericardial effusion? (Pericardial effusion is fluid around the heart.)", 
                    ("Yes", "No"), index=1)

if checkbox_val == "Yes" and uploaded_file is not None:
    if st.button("Submit"):

        # Validate inputs
        try:
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
            st.dataframe(test_data_df_updated)

            # Display the Plotly Confusion Matrix
            html_data = read_from_json('model/RandomForest_CM.json')
            st.plotly_chart(html_data, use_container_width=True, template='plotly')

            # Display Discrimination Threshold - Random Forest
            #html_data = read_from_json('model/rf_disc_threshold.json')
            #st.plotly_chart(html_data, use_container_width=True, template='plotly')

            # Display Discrimination Threshold - Random Forest - Alternative
            HtmlFile = open('model/rf_disc_threshold.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height = 600, width=1000)

            # Display a ROC-AUC comparison plot 
            HtmlFile = open('model/roc_auc_comparison.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height = 800, width=1000)

            # Display an image of a guy having an heart attack
            st.image('model/heart_attack.png', caption='Heart Attack')

        except Exception as e:
            st.error(f"An error occurred: {e}")

elif checkbox_val == "No":
    if st.button("Submit"):

        df = pd.DataFrame(data = {
            'age-at-heart-attack': age,
            'pericardial-effusion': pericardial_effusion,
            'fractional-shortening': frac_short,
            'epss': epss,
            'lvdd': lvdd,
            'wall-motion-index': wall_motion_index,
            'age_lvdd_interaction': age * lvdd,
            'wall-motion-index_lvdd_interaction': wall_motion_index * lvdd}, index=[0]
        )

        # Display the Chosen Values
        chosen_val = df[['age-at-heart-attack',
            'pericardial-effusion',
            'fractional-shortening',
            'epss',
            'lvdd',
            'wall-motion-index']].T
        chosen_val.columns = ['Your Values']
        st.dataframe(chosen_val)

        # Define age groups
        bins = [0, 55, 65, 100]  # Define the age boundaries for each group
        labels = ['0-55', '55-65', '65-100']  # Define labels for each group

        # Create a new column for age groups
        df['age_group'] = pd.cut(df['age-at-heart-attack'], bins=bins, labels=labels, right=False)

        # Convert 'Yes' and 'No' to 0 and 1
        df['pericardial-effusion'] = np.where(df['pericardial-effusion'] == 'Yes', 1, 0)

        # Predictions
        filename = 'model/LogisticRegression.sav'
        lg_pipe = pickle.load(open(filename, 'rb'))

        # Predictions by Random Forest 
        y_pred = lg_pipe.predict(df)
        probabilities = lg_pipe.predict_proba(df)
        max_prob = np.max(lg_pipe.predict_proba(df), axis=1)[0] * 100  

        # Returns the result
        st.success(f"This patient {map_number_to_class(y_pred[0])} with a probability of {max_prob:.2f}%.")

        classes = ["Won't Survive", "Survive"]
        # Bar chart showing probabilities for each class
        df_prob = pd.DataFrame({'Class': classes, 'Probability': probabilities[0]})
        chart = alt.Chart(df_prob).mark_bar().encode(
            x='Probability',
            y=alt.Y('Class', sort='-x')
        ).properties(
            width=500,
            height=200
        )
        st.altair_chart(chart, use_container_width=True)

        # Display the Plotly Confusion Matrix
        html_data = read_from_json('model/RandomForest_CM.json')
        st.plotly_chart(html_data, use_container_width=True, template='plotly')

        # Display Discrimination Threshold - Random Forest
        #html_data = read_from_json('model/rf_disc_threshold.json')
        #st.plotly_chart(html_data, use_container_width=True, template='plotly')

        # Display Discrimination Threshold - Random Forest - Alternative
        HtmlFile = open('model/rf_disc_threshold.html', 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height = 600, width=1000)

        # Display a ROC-AUC comparison plot 
        HtmlFile = open('model/roc_auc_comparison.html', 'r', encoding='utf-8')
        source_code = HtmlFile.read() 
        components.html(source_code, height = 800, width=1000)

        # Display an image of a guy having an heart attack
        st.image('model/heart_attack.png', caption='Heart Attack')
