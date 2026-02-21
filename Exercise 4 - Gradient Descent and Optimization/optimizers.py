###########################################################################
#               Physics-Informed Machine Learning                         #
#                             WS 2025                                     #
#                                                                         #
#                      Exercise 4 - Template                              #
#                                                                         #
# NOTICE: Sharing and distribution of any course content, other than      #
# between individual students registered in the course, is not permitted  #
# without permission.                                                     #
#                                                                         #
###########################################################################

import numpy as np

EPS = 1e-8  # global variable, available to all function in this script


class Gradient_Descent():
    def __init__(self, compute_gradient, lr):
        """
        Class implements the iterative optimization method - Gradient Descent.

        Parameters:

            compute_gradient: function takes point of dim= D returns gradient of dim= D
                function that computes the gradient of an objective function optimized for.

            lr: float
                learning rate (alpha) of an update step.
        """
        self.compute_gradient = compute_gradient
        self.lr = lr

    def step(self, point):
        ##############
        ##############
        gradient = self.compute_gradient(point)
        point = point - self.lr * gradient
        ##############
        ##############
        return point


class Newton_Method_GD(Gradient_Descent):
    def __init__(self, compute_gradient, lr, compute_hessian):
        """
        Class implements the iterative optimization method - Newton's Method.

        Parameters:

            compute_gradient: function, takes point of shape= D returns gradient of dim= D
                function that computes the gradient of an objective function optimized for.

            lr: float
                learning rate (alpha) of an update step.

            compute_hessian: function, takes point of dim= D returns matrix of shape= [D,D]
                function that computes the hessian mat of an objective function optimized for
        """
        super().__init__(compute_gradient, lr)
        self.compute_hessian = compute_hessian

    def step(self, point):
        ##############
        ##############
        gradient = self.compute_gradient(point)
        hessian = self.compute_hessian(point)
        hessian_inv = np.linalg.inv(hessian)
        point = point - self.lr * hessian_inv @ gradient
        ##############
        ##############
        return point


class RMSProp(Gradient_Descent):
    def __init__(self, compute_gradient, lr, beta_2=0.9):
        """
        Class implements the iterative optimization method - RMSProp

        Parameters:

                beta_2: float, beta_2 parameter in RMSProp equation
                v: float, v parameter in RMSProp equation
        """
        super().__init__(compute_gradient, lr)
        self.beta_2 = beta_2
        self.v = .1


    def step(self, point):
        ##############
        ##############
        gradient = self.compute_gradient(point)
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient * gradient
        point = point - self.lr * gradient / (np.sqrt(self.v) + EPS)
        ##############
        ##############
        return point


class ADAM(RMSProp):
    """
    Class implements the iterative optimization method - ADAM
    """

    def __init__(self, compute_gradient, lr, beta_2=0.999, beta_1=0.9):
        """
        Class implements the iterative optimization method - RMSProp

        Parameters:

                beta_1: float, beta_1 parameter in ADAM equation
        """
        super().__init__(compute_gradient, lr, beta_2=beta_2)
        self.m = .1
        self.beta_1 = beta_1
        self.i = 0


    def step(self, point):
        ##############
        ##############
        self.i += 1
        gradient = self.compute_gradient(point)
        
        self.m = self.beta_1 * self.m + (1 - self.beta_1) * gradient
        self.v = self.beta_2 * self.v + (1 - self.beta_2) * gradient * gradient
        
        m_hat = self.m / (1 - self.beta_1 ** self.i)
        v_hat = self.v / (1 - self.beta_2 ** self.i)
        
        point = point - self.lr * m_hat / (np.sqrt(v_hat) + EPS)
        ##############
        ##############
        return point