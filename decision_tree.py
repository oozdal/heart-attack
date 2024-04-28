from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def decision_tree(X_train, y_train, X_test, y_test):
    """
    Decision Tree Classifier
    """

    print("\n")
    print("============================= Decision Tree Classifier ==================================")
    
    # Define the parameter grid
    param_grid = {'criterion': ['gini', 'entropy', 'log_loss'], 
              'max_depth': [None, 5, 10, 20],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

    # Initialize Decision Tree classifier
    dt = DecisionTreeClassifier(random_state=42)
    
    # Initialize GridSearchCV with the classifier and parameter grid
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, scoring='roc_auc', cv=5, n_jobs=-1)
    
    # Perform grid search on training data
    grid_search.fit(X_train, y_train)
    
    # Get the scores from the grid search
    scores = grid_search.cv_results_['mean_test_score']
    
    # Get the best parameters and best accuracy
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Initialize DecisionTreeClassifier with the best parameters
    best_classifier = DecisionTreeClassifier(**best_params, random_state=42)
    
    # Fit the classifier on the training data
    best_classifier.fit(X_train, y_train)
    
    # Predict the target variable for the test data
    y_pred = best_classifier.predict(X_test)
    
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred)
    
    # Output the best parameters and evaluation metrics
    print("Best Parameters:", best_params)
    print("F1 Macro Score: {:.2f} %".format(best_score*100))
    print("Accuracy: {:.2f} %".format(accuracy*100))
    
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