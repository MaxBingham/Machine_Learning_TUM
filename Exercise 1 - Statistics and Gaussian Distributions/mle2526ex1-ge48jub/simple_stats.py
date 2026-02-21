###########################################################################
#               Physics-Informed Machine Learning                         #
#                             SS 2023                                     #
#                                                                         #
#                     Exercise 1 - Template                               #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import numpy as np
import matplotlib.pyplot as plt

def readfile(filename, delimiter):
    with open(filename, 'r') as file:
        lines = file.readlines()  
        split_lines = [line.strip().split(delimiter) for line in lines]
    # Convert strings to float numbers
    data = np.array(split_lines, dtype=float) 
    
    return data


def compute_mean(x):
    E = None  # E stands for the expected value
    E = np.sum(x) / x.shape[0]
    return E


def compute_variance(x):
    sigma2 = None  
    mean_x = compute_mean(x)
    mean_x_squared = compute_mean(x**2)
    sigma2 = mean_x_squared - mean_x**2
    return sigma2


def compute_covariance(data):
    N, D = data.shape
    cov = np.zeros((D, D))
    for i in range(D):
        for j in range(D):
            xi = data[:, i]
            xj = data[:, j]
            cov[i, j] = np.mean(xi * xj) - np.mean(xi) * np.mean(xj)
    return cov


def main():
    # Read the dice data
    data = readfile("dice.txt", ",")
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"First 5 rows:\n{data[:5]}")
    
    # Create histogram for all three dice
    plt.figure(figsize=(10, 6))
    for i in range(data.shape[1]):
        plt.hist(data[:, i], bins=np.arange(1, 8) - 0.5, alpha=0.7, 
                label=f"Dice {i+1}", edgecolor='black')
    plt.xlabel("Dice Outcome")
    plt.ylabel("Frequency")
    plt.title("Distribution of Dice Rolls")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    # Test mean function
    print("\n=== MEAN TEST ===")
    for i in range(data.shape[1]):
        my_mean = compute_mean(data[:, i])
        numpy_mean = np.mean(data[:, i])
        print(f"Dice {i+1}: My function = {my_mean:.6f}, NumPy = {numpy_mean:.6f}")
    
    # Test variance function
    print("\n=== VARIANCE TEST ===")
    for i in range(data.shape[1]):
        my_var = compute_variance(data[:, i])
        numpy_var = np.var(data[:, i])
        print(f"Dice {i+1}: My function = {my_var:.6f}, NumPy = {numpy_var:.6f}")
    
    # Test covariance function
    print("\n=== COVARIANCE TEST ===")
    my_cov = compute_covariance(data)
    numpy_cov = np.cov(data, rowvar=False)
    
    print("My covariance matrix:")
    print(my_cov)
    print("\nNumPy covariance matrix:")
    print(numpy_cov)

    return 0


if __name__ == "__main__":
    main()
