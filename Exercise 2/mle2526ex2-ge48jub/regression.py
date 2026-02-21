###########################################################################
#               Physics-Informed Machine Learning                         #
#                             SS 2023                                     #
#                                                                         #
#                     Exercise 2 - Template                               #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import numpy as np


def polynomial_basis_matrix(X, degree):
    """
    Parameters
    ----------
    X : array [N,1]
        input array
    degree : int
        degree of the polynomial

    Returns
    -------
    Phi : array [N, degree + 1]
        polynomial basis matrix
    """
    Phi = None
    X = X.flatten()

    Phi = np.vander(X, degree+1, increasing=True)
    return Phi




def gaussian_basis_matrix(X, parameters):
    """
    Parameters
    ----------
    X : array [N,1]
        input.
    parameters : list
        each element is tuple of the mean and standard deviation.

    Returns
    -------
    Phi : array [N, len(list_parameters)]
        gaussian basis matrix
    """
    
    # FIX: Prepare X for vectorized calculations (flatten to 1D)
    X_flat = X.flatten()
    N = len(X_flat)
    n_basis = len(parameters)


    Phi = np.zeros((N, n_basis))

    for i in range(len(parameters)):
        
        mu = parameters[i][0]
        sigma = parameters[i][1]


        gaussian_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((X_flat - mu) / sigma)**2)
        
        Phi[:, i] = gaussian_values
        
    return Phi


class LeastSquares(object):
    def __init__(self, basis_function):
        """
        Parameters
        ----------
        basis_function : lambda function
            Mapping of input values
        """
        self.weights = None
        self.basis_function = basis_function

    def fit(self, X, y):
        """Fits ordinary least squares model to the
        given data by finding the appropriate weights.
        
        Parameters
        ----------
        X : array, shape [N, D]
            input array (before applying a basis function transformation!)
        y : array, shape [N]
            (true) outputs corresponding to provided inputs

        """
        Phi = None
        # X=X.flatten()
        # y=y.flatten()
        # Apply the basis function to input X to get the design matrix
        Phi = self.basis_function(X)
        PhiT = np.transpose(Phi)
        PhiInv = np.linalg.inv(PhiT @ Phi)
    
        self.weights = PhiInv @ PhiT @ y



    def predict(self, X):
        """Generates predictions for the given samples.
        
        Parameters
        ----------
        X : array, shape [N, D]
            input array (before applying a basis function transformation!)

        Returns
        -------
        y_pred : array, shape [N]
            predicted outputs corresponding to provided inputs
            
        """
        y_pred = None
        if self.weights is not None:
            Phi = self.basis_function(X) 
            
            y_pred = Phi @ self.weights  # Use matrix multiplication, not element-wise
        return y_pred

        


def MSE(y_true, y_pred):
    """Computes Mean Squared Error
    between true and predicted outputs.
        
    Parameters
    ----------
    y_true : array, shape [N]
        true outputs
    y_pred : array, shape [N]
        predicted outputs
        
    Returns
    -------
    mse : float
        Mean Squared Error
        
    """
    mse = 0.0  # None is its own value, Zero means it doesnt have a value
    N = len(y_true)  # Get number of samples

    mse = float(np.mean((y_true - y_pred) ** 2))

    return mse 


def evaluate(path_training1="training_dataset_1.npy", path_training2="training_dataset_2.npy",
             path_test1="test_dataset_1.npy", path_test2="test_dataset_2.npy"):
    # Please do not change the input parameters of evaluate() and make sure you return
    # the models and training/test error values in the order given in the template, otherwise the
    # testing framework will not match them to the corresponding expected values and the test will fail.

    import matplotlib.pyplot as plt

    # loading datasets
    training_dataset_1, test_dataset_1 = np.load(path_training1), np.load(path_test1)
    X1_train, y1_train = np.expand_dims(training_dataset_1[0, :], -1) / 1000, training_dataset_1[1, :] / 1000
    # the values in the dataset were too large, so I scaled them down by 1000 here
    X1_test, y1_test = np.expand_dims(test_dataset_1[0, :], -1) / 1000, test_dataset_1[1, :] / 1000
    # print(X_train.shape)
    # print(X_test.shape)

    # visualize training dataset
    plt.scatter(X1_train, y1_train, s=20, marker='x')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Dataset1")
    # plt.show()  # Commented out for testing

    # polynomial:
    polynomial_degree = 7  # Increased from 2 to 5 for better fit
    polynomial_basis_func = lambda x: polynomial_basis_matrix(x, polynomial_degree)
    model1 = LeastSquares(polynomial_basis_func)

    # gaussian:
    # list_params = [(3, 1), (4, 2), (10, 3)]
    # gaussian = lambda x: gaussian_basis_matrix(x, list_params)
    # model1 = Least_Squares(gaussian)

    model1.fit(X1_train, y1_train)

    # compute predictions on training set 1:
    predictions1_train = None

    predictions1_train = model1.predict(X1_train)


    fig = plt.figure()

    plt.scatter(np.concatenate([X1_train, X1_test]), np.concatenate([y1_train, y1_test]), s=20, marker='x')
    # use np.concatenate or np.hstack if you want to stack numpy arrays together

    sorted_indecies = np.argsort(X1_train, axis=0)
    sorted_X = np.squeeze(X1_train[sorted_indecies])
    sorted_predictions = np.squeeze(predictions1_train[sorted_indecies])
    plt.plot(sorted_X, sorted_predictions, 'r', label='fitted function')
    plt.legend(loc='lower right')
    plt.title('Regression')
    # plt.show()  # Commented out for testing

    # compute mse on training set 1:
    MSE_error1_train = None

    MSE_error1_train = MSE(y1_train, predictions1_train)

    # compute mse on test set 1:
    MSE_error1_test = None

    predictions1_test = model1.predict(X1_test)
    MSE_error1_test = MSE(y1_test, predictions1_test)

    # moving on to dataset 2

    # loading datasets
    training_dataset_2, test_dataset_2 = np.load(path_training2), np.load(path_test2)
    X2_train, y2_train = np.expand_dims(training_dataset_2[0, :], -1), training_dataset_2[1, :]
    X2_test, y2_test = np.expand_dims(test_dataset_2[0, :], -1), test_dataset_2[1, :]

    plt.scatter(X2_train, y2_train, s=20, marker="x")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Dataset2")
    plt.show()  # Commented out for testing

    polynomial_degree = 7
    polynomial_basis_func = lambda x: polynomial_basis_matrix(x, polynomial_degree)
    model2 = LeastSquares(polynomial_basis_func)

    # gaussian:
    # list_params = [(3, 1), (4, 2), (10, 3)]
    # gaussian = lambda x: gaussian_basis_matrix(x, list_params)
    # model2 = Least_Squares(gaussian)

    # fit model2 to training set 2:

    model2.fit(X2_train, y2_train)

    # compute predictions on training set 2:
    predictions2_train = None

    predictions2_train = model2.predict(X2_train)

    plt.scatter(X2_train, y2_train, s=20, marker='x', color='b')
    sorted_indecies = np.argsort(X2_train, axis=0)
    sorted_X = np.squeeze(X2_train[sorted_indecies])
    sorted_predictions = np.squeeze(predictions2_train[sorted_indecies])
    plt.plot(sorted_X, sorted_predictions, 'r', label='fitted function')
    plt.legend(loc='lower right')
    plt.title('Regression')
    plt.show()  # Commented out for testing

    # compute mse on training set 2:
    MSE_error2_train = None

    MSE_error2_train = MSE(y2_train, predictions2_train)

    # compute mse on test set 2:
    MSE_error2_test = None

    predictions2_test = model2.predict(X2_test)
    MSE_error2_test = MSE(y2_test, predictions2_test)

    return model1, MSE_error1_train, MSE_error1_test, model2, MSE_error2_train, MSE_error2_test


def main():
    # You can uncomment evaluate for running it on your own computer, or check the other functions here
    results = evaluate()
    model1, MSE_error1_train, MSE_error1_test, model2, MSE_error2_train, MSE_error2_test = results
    
    print("=== EVALUATION RESULTS ===")
    print(f"Dataset 1 - Training MSE: {MSE_error1_train:.6f}")
    print(f"Dataset 1 - Test MSE: {MSE_error1_test:.6f}")
    print(f"Dataset 2 - Training MSE: {MSE_error2_train:.6f}")
    print(f"Dataset 2 - Test MSE: {MSE_error2_test:.6f}")

    pass

if __name__ == "__main__":
    main()
