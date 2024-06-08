# Two-year Survival Predictor After Having a Heart Attack (Web App)

The Two-year Survival Predictor After Having a Heart Attack is an application designed to predict patient survival rates following a heart attack, focusing on helping healthcare professionals in making informed decisions and delivering personalized care. 
This project leverages data analysis techniques and machine learning algorithms to provide accurate predictions.

# Description

This project aimed to develop a predictive model for assessing whether patients survived for at least two years following a heart attack. 
The methodology commenced with comprehensive exploratory data analysis to elucidate the distributions of the provided features. 
Subsequently, feature engineering was conducted, including the creation of a new categorical variable, 'age group,' derived from the 'age-at-heart-attack' column. 
This involved categorizing patients into three distinct groups based on age ranges: 0-55, 55-65, and 65-100. 
Additionally, two interaction terms were generated to capture potential synergies between features: 'age_lvdd_interactions' and 'wall-motion-index_lvdd_interactions,' created by multiplying 'age-at-heart-attack' with 'lvdd,' and 'wall-motion-index' with 'lvdd,' respectively.

Following this, patients who either survived beyond a two-year period or succumbed within the same timeframe were identified, forming the labeled dataset for supervised classification. 
However, a significant class imbalance was encountered, with only one patient labeled as 'survived' and 60 patients labeled as 'not-survived.' 
To address this, unsupervised clustering was initially employed on the unlabelled data, consisting of patients who experienced a heart attack within two years and remained alive. 
Labels were assigned to the clustered data and incorporated into the labeled dataset to augment the number of positive samples. 
Despite this integration, the target distribution remained imbalanced, with 82 negative samples and 8 positive samples. 
To rectify this, oversampling techniques, specifically the ADASYN algorithm, were employed, effectively balancing the target distribution.

Following data preprocessing, the dataset was prepared for model training by applying OneHotEncoding to categorical variables, performing feature standardization, and splitting the dataset into training and test sets. 
Hyperparameters for eight established classification algorithms were optimized using GridSearch, maximizing the ROC-AUC metric for each algorithm with five-fold cross-validation. 
The ROC-AUC metric was favored over accuracy due to its ability to evaluate classifier performance across various thresholds, making it more suitable for imbalanced datasets.

From the eight models, the Random Forest and XGBoost Classifier were selected for making predictions on the provided echocardiogram.test data. 
Predictions and prediction probabilities of these models were recorded as new columns. 
Agreement between the predictions of both algorithms and their close prediction probabilities indicated the success of the analysis. 
Finally, SHAP values were examined to better understand the decisions made by both algorithms, revealing fractional-shortening and wall-motion-index_lvdd_interaction as the two most influential features. 
The prominence of the added interaction terms in the SHAP charts underscored the effectiveness of feature engineering in this analysis.

To enhance the dataset, I would suggest adding features such as existing medical conditions (diabetes, hypertension, previous heart conditions), medication details post-heart attack, lifestyle factors (smoking, alcohol, diet, exercise), genetic information, 
and relevant biomarkers (troponin levels, BNP, CRP). Enriching the dataset with these factors could significantly improve the accuracy of predictive models for forecasting patient survival rates post-heart attack.

# Usage, Deployment & Streamlit User Interface

The application is deployed and accessible at: [Two-year Survival Predictor Application (After Heart Attack)](https://heart-attack-j6dh.onrender.com/)

Warning: Free Instance Spin-Down Delay

Please note that the free instance provided by Render may experience spin-down due to inactivity. This could result in delays of 50 seconds or more when processing requests. Please be patient while your web browser tries to load the page.

# Screenshots

If you have an `echocardiogram.test` file, You can upload your file to obtain predictions as demonstrated below. Once you click 'Yes' for the following question, a file uploader UI component will appear. You can then use this UI component to drag and drop your .csv file.

![Screenshot 2024-06-08 150239](https://github.com/oozdal/heart-attack/assets/34719109/02781c28-cc9c-416c-a47d-1d1e1deb42ea)

The predictions and their associated probabilities will be appended as additional columns to your `echocardiogram.test` file as demonstrated below.

![Screenshot 2024-06-08 151151](https://github.com/oozdal/heart-attack/assets/34719109/309409b6-e460-4f26-9162-6da32f38dcaf)

If you do not have an `echocardiogram.test` file, you can answer the following questions to obtain your predictions as demonstrated below.

![Screenshot 2024-06-08 145605](https://github.com/oozdal/heart-attack/assets/34719109/862226ab-265a-4d66-a045-13125bed5a7c)

Once you click `Submit`, you will see your selected values along with your prediction. Additionally, a plot displaying the probabilities of the predictions will be shown.

![Screenshot 2024-06-08 150011](https://github.com/oozdal/heart-attack/assets/34719109/6742f01c-e89e-407e-b1f4-c48d67da65ae)

# Interactive Plotly Figures

The Heart Attack App also provides highly interactive Plotly figures that facilitate the extraction of insights.

![Interactive ROC AUC Curve](https://github.com/oozdal/heart-attack/assets/34719109/218d118f-71d0-4de9-86bf-60efba98e3dd)

![Interactive Disc Plot](https://github.com/oozdal/heart-attack/assets/34719109/d0783615-ecd9-45ea-835e-338785a6fc9f)

