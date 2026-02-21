###########################################################################
#               Physics-Informed Machine Learning                         #
#                             SS 2025                                     #
#                                                                         #
#                  Exercise 4 - Cross Validation                          #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import numpy as np
from sklearn.model_selection import KFold

try:
    from .common import polynomial_basis_matrix, PolynomialBasis
except ImportError:
    from common import polynomial_basis_matrix, PolynomialBasis


class LeastSquares(object):
    def __init__(self, basis_function):
        """
        Parameters
        ----------
        basis_function : callable
            Maps inputs X [N,1] to design matrix Phi [N,M]
        """
        self.basis_function = basis_function
        self.weights = None

    def fit(self, X, y):
        """
        Least Squares fit (closed form via lstsq; stable)
        Parameters
        ----------
        X : np.array [N,1]
        y : np.array [N]
        """
        Phi = self.basis_function(X) 
        y = np.asarray(y).reshape(-1, 1)
        w, *_ = np.linalg.lstsq(Phi, y, rcond=None)
        self.weights = w.reshape(-1)

    def predict(self, X):
        """
        Parameters
        ----------
        X : np.array [N,1]
        Returns
        -------
        y_pred : np.array [N]
        """
        if self.weights is None:
            return None
        Phi = self.basis_function(X)
        y_pred = Phi @ self.weights
        return np.asarray(y_pred).reshape(-1)


def MSE(y_true, y_pred):
    """
    Mean Squared Error
    """
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return float(np.mean((y_true - y_pred) ** 2))


def cross_validation(X, y, model, K=10, random_state=42, shuffle=True):
    """
    K-fold cross-validation for a given model (basis-function LS).

    Parameters
    ----------
    X : np.array [N,1]
    y : np.array [N]
    model : LeastSquares
        A prototype model with the desired basis_function set. A fresh
        model will be constructed inside each fold using this basis.
    K : int
        number of folds (default: 10)
    random_state : int
        seed for deterministic shuffling
    shuffle : bool
        whether to shuffle before splitting

    Returns
    -------
    loss : float
        average validation MSE across folds
    """
    if shuffle:
        kf = KFold(n_splits=K, shuffle=shuffle, random_state=random_state)
    else:
        kf = KFold(n_splits=K, shuffle=shuffle)

    # TODO [Task 1]: implement K-fold cross-validation
    # - for each split, build a fresh model: LeastSquares(model.basis_function)
    # - fit on the training fold, predict on the validation fold
    # - accumulate MSE on the validation fold
    # - return the average over all folds

    total = 0.0
    count = 0
    for train_idx, val_idx in kf.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        ####### TODO #######
        # Create a fresh model with the same basis function
        fold_model = LeastSquares(model.basis_function)
        
        # Fit on the training fold
        fold_model.fit(X_tr, y_tr)

        # Predict on validation fold and compute MSE
        y_pred = fold_model.predict(X_val)
        fold_mse = MSE(y_val, y_pred)
        ####################

        total += fold_mse
        count += 1

    loss = total / max(count, 1)
    return loss


def model_selection(X, y, degrees, K=10, random_state=42, shuffle=True):
    """
    Select the best polynomial degree using K-fold cross-validation.

    Parameters
    ----------
    X : np.array [N,1]
    y : np.array [N]
    degrees : list[int]
        candidate polynomial degrees (e.g., [1,2,3,4,5,6,7,8,9,10])
    K, random_state, shuffle : see cross_validation()

    Returns
    -------
    selected_model : LeastSquares
        model trained on the full data with the best degree
    selected_degree : int
        chosen degree (argmin CV loss)
    losses : dict
        mapping degree -> CV loss
    """
    losses = {}

    # TODO [Task 2]: evaluate each degree with CV and pick the best
    # Steps:
    # - for each d in degrees:
    #     * create the PolynomialBasis
    #     * compute cv_loss = cross_validation(X, y, model, K, random_state, shuffle)
    #     * store in losses[d] = cv_loss
    # - find degree with minimal loss
    # - fit selected_model on all (X,y) with that degree and return it

    for d in degrees:
        ####### TODO #######
        # Create polynomial basis for degree d
        basis = PolynomialBasis(d)
        
        # Create prototype model
        prototype_model = LeastSquares(basis)
        
        # Compute cross-validation loss
        cv_loss = cross_validation(X, y, prototype_model, K, random_state, shuffle)
        
        # Store the loss
        losses[d] = cv_loss
        ####################

    # select the best degree
    selected_degree = min(losses, key=losses.get)

    ###### TODO ######
    # Create final model with selected degree and fit on all data
    selected_basis = PolynomialBasis(selected_degree)
    selected_model = LeastSquares(selected_basis)
    selected_model.fit(X, y)
    ##################

    return selected_model, selected_degree, losses


def main():    
    rng = np.random.default_rng(0)
    N = 40
    X = rng.uniform(0.0, 1.0, size=(N, 1))
    y = np.sin(2 * np.pi * X[:, 0]) + rng.normal(0, 0.1, size=N)

    degrees = list(range(1, 11))
    _, d_star, losses = model_selection(X, y, degrees, K=10, random_state=0)

    print("Selected degree:", d_star)
    print("CV losses (degree -> MSE):")
    for d in degrees:
        print(f"{d:2d}: {losses[d]:.5f}")


if __name__ == "__main__":
    main()
