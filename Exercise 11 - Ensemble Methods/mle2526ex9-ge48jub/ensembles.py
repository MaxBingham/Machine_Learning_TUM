import numpy as np
from sklearn.ensemble import RandomForestRegressor, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score

def train_rf_regressor(X_train, y_train, n_estimators=100, max_features='sqrt'):
    """
    Task 1: Bagging with Random Forest.
    - Initialize a RandomForestRegressor with the given n_estimators and max_features.
    - Use random_state=42 for reproducibility.
    - Return the trained model object.
    """

    ######
    #Clarify mental model: p - selects the amount of features considered at each split. 
    #Therefore data is not online split into amounts, but also in terms of features
    ######
    
    model = None
    
    model = RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)
    model = model.fit(X_train, y_train)

    return model
#--> Now sampling data with p and n 

def get_oob_score(X_train, y_train, n_estimators=100):
    """
    Task 2: Out-of-Bag Evaluation.
    - Initialize a RandomForestRegressor with oob_score=True and random_state=42.
    - Fit the model on the training data.
    - Return the oob_score_ attribute.
    """
    #Evaluate model with out of bag eval - points never sampled by bootstrap
    score = 0.0
    
    score = RandomForestRegressor(n_estimators=n_estimators, oob_score=True, random_state=42)
    score = score.fit(X_train, y_train)

    return score.oob_score_

def get_top_features(model, feature_names):
    """
    Task 3: Feature Importance.
    - Extract feature_importances_ from the trained model.
    - Return a list of tuples [(feature_name, importance), ...] sorted by importance descending.
    """
    importance = model.feature_importances_
    importance_pairs = list(zip(feature_names, importance))

    sorted_features = sorted(importance_pairs, key = lambda x: x[1], reverse=True)
    return sorted_features

def train_adaboost_classifier(X_train, y_train, n_estimators=50):
    """
    Task 4: Boosting with AdaBoost.
    - Initialize a DecisionTreeClassifier with max_depth=1 (a Decision Stump).
    - Use this stump as the base estimator for AdaBoostClassifier.
    - Use random_state=42.
    - Return the trained model object.
    """
    
    model = None
    #week classifiere 
    estimators = DecisionTreeClassifier(max_depth=1)
    #compute parameters + weights and create a weighted average 
    model = AdaBoostClassifier(estimator=estimators, n_estimators=n_estimators, random_state=42)
    model = model.fit(X_train, y_train)
    
    return model

def get_staged_scores(model, X_test, y_test):
    """
    Task 5: Staged Prediction Analysis.
    - Use the staged_score(X_test, y_test) method of the AdaBoost model.
    - Convert the generator to a list and return it.
    """
    
    stage = model.staged_score(X_test, y_test)
    staged_results = list(stage)
    return staged_results
    

def train_gradient_boosting(X_train, y_train, n_estimators=100):
    """
    Task 6: Gradient Boosting.
    - Initialize and train a GradientBoostingClassifier with random_state=42.
    - Return the trained model object.
    """
    model = None
    model = GradientBoostingClassifier(n_estimators=n_estimators, random_state=42)
    model = model.fit(X_train, y_train) 

    return model