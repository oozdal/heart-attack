from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import ADASYN
import numpy as np


def logistic_regression(X_train, y_train, X_test, y_test):
    """
    Logistic Regression Classifier
    """

    print("\n")
    print("============================= Logistic Regression Classifier ==================================")
    
    # Define which columns to scale and which to leave as is
    numeric_features = ['age-at-heart-attack',
        'fractional-shortening', 'epss', 'lvdd', 'wall-motion-index',
        'age_lvdd_interaction', 'wall-motion-index_lvdd_interaction']

    categorical_features = ['age_group', 'pericardial-effusion']

    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='error'), categorical_features)
        ])

    # Oversampling using ADASYN
    oversampler = ADASYN(
        sampling_strategy = 'auto', # samples only the minority class
        random_state = 42, # for reproducibility
        n_neighbors = 3,
        n_jobs = 4
    )

    # creating a pipe using the make_pipeline method
    pipe = Pipeline([('Scaler', preprocessor), ('oversampler', oversampler), ('clf', LogisticRegression())])

    # Define the parameter grid
    param_grid = {'clf__C': [0.001, 0.01, 0.1, 1, 10, 100],
              'clf__penalty': ['l2'],
              'clf__solver' : ['liblinear']}
    
    # Initialize GridSearchCV with the classifier and parameter grid
    grid_search = GridSearchCV(estimator=pipe, param_grid=param_grid, scoring='recall_micro', cv=5, n_jobs=-1)
    
    # Perform grid search on training data
    grid_search.fit(X_train, y_train)
    
    # Get the scores from the grid search
    scores = grid_search.cv_results_['mean_test_score']
    
    # Get the best parameters and best accuracy
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Access the best model 
    best_classifier = grid_search.best_estimator_
    
    # Predict the target variable for the test data
    y_pred = best_classifier.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = balanced_accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    
    # Output the best parameters and evaluation metrics
    print("Best Parameters:", best_params)
    print("Best Score: {:.2f} %".format(best_score*100))
    print("Balanced Accuracy: {:.2f} %".format(accuracy*100))
    
    # Calculate the standard deviation of scores
    std_dev = np.std(scores)*100
    
    # Output the standard deviation
    print("Standard Deviation of Cross Validation Scores: {:.2f} %".format(std_dev))
    
    # Confusion Matrix
    print("Confusion Matrix:\n", conf_matrix)
    
    # Classification Report
    print("Classification Report:\n", clf_report)

    # Return the best parameters and best estimator
    return grid_search.best_params_, grid_search.best_estimator_

