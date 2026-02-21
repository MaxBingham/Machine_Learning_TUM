from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def build_voting_ensemble():
    """
    Task 7: Building a Voting Ensemble.
    - Create at least 3 base classifiers: RandomForest, SVC (with probability=True), and KNeighborsClassifier.
    - Combine them into a VotingClassifier using voting='soft'.
    - Return the unfitted ensemble object.
    """
    
    model = None
    model1 = RandomForestClassifier(n_estimators=50, random_state=42)
    model2 = SVC(probability=True)
    model3 = KNeighborsClassifier(n_neighbors=5)

    estimators = [('first', model1), ('second', model2), ('third', model3)] 
    model = VotingClassifier(estimators=estimators, voting='soft')

    return model

def build_best_stacked_model():
    """
    Task 8: Stacking Playground.
    - Define a list of base estimators and a final_estimator (e.g., LogisticRegression).
    - Use StackingClassifier to combine them.
    - Wrap the ensemble in a Pipeline with a StandardScaler if needed.
    - Return the unfitted model object.
    """

    model = None

    model1 = RandomForestClassifier(n_estimators=50, random_state=42)
    model2 = SVC(probability=True)
    model3 = KNeighborsClassifier(n_neighbors=5)

    base_estimators = [('first', model1), ('second', model2), ('third', model3)] 
    final_estimator = LogisticRegression()

    combined = StackingClassifier(estimators=base_estimators, final_estimator=final_estimator)

    model = Pipeline([
    ('name1', StandardScaler()),
    ('name2', combined)
    ])

    return model