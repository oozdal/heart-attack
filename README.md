# Survival Predictor After Having a Heart Attack

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