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


class Convex_Func():
    @staticmethod
    def eval(point):
        x, y = point[0], point[1]
        func = x ** 2 + y ** 2
        return func

    @staticmethod
    def gradient(point):
        x, y = point[0], point[1]
        dx = 2. * x
        dy = 2. * y
        grad = np.array([dx, dy])
        return grad

    @staticmethod
    def hessian(point):
        x, y = point[0], point[1]
        hessian = np.array([[2., 0], [0, 2.]])
        return hessian


class Beale_Func():
    @staticmethod
    def eval(point):
        x, y = point[0], point[1]
        func = None
        ##############
        ##############
        term1 = (1.5 - x + x*y)**2
        term2 = (2.25 - x + x*y**2)**2
        term3 = (2.625 - x + x*y**3)**2
        func = term1 + term2 + term3
        ##############
        ##############
        return func

    @staticmethod
    def gradient(point):
        x, y = point[0], point[1]
        grad = None
        ##############
        ##############
        u1 = 1.5 - x + x*y
        u2 = 2.25 - x + x*y**2
        u3 = 2.625 - x + x*y**3
        
        dx = 2*u1*(-1 + y) + 2*u2*(-1 + y**2) + 2*u3*(-1 + y**3)
        dy = 2*u1*x + 2*u2*(2*x*y) + 2*u3*(3*x*y**2)
        
        grad = np.array([dx, dy])
        ##############
        ##############
        return grad
    
    @staticmethod
    def hessian(point):
        x, y = point[0], point[1]
        h11 = 2 * (y ** 6 + y ** 4 - 2 * y ** 3 - y ** 2 + 2 * y + 3)
        h12 = 4 * x * (3 * y ** 5 + 2 * y ** 3 - 3 * y ** 2 - y + 1) + 15.75 * y ** 2 + 9 * y - 3
        h21 = 2 * (12 * y ** 5 + 8 * y ** 3 - 12 * y ** 2 - 4 * y + 4) + 15.75 * y ** 2 + 9 * y - 3
        h22 = 2 * x * (x * (15 * y ** 4 + 6 * y ** 2 - 6 * y - 1) + 15.75 * y + 4.5)
        hessian = np.array([[h11, h12], [h21, h22]])
        return hessian



class Himmelblaus_Func():
    @staticmethod
    def eval(point):
        x, y = point[0], point[1]
        func = None
        ##############
        ##############
        func = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
        ##############
        ##############
        return func

    @staticmethod
    def gradient(point):
        x, y = point[0], point[1]
        grad = None
        ##############
        ##############
        dx = 2*(x**2 + y - 11)*2*x + 2*(x + y**2 - 7)
        dy = 2*(x**2 + y - 11) + 2*(x + y**2 - 7)*2*y
        
        grad = np.array([dx, dy])
        ##############
        ##############
        return grad

    @staticmethod
    def hessian(point):
        x, y = point[0], point[1]
        h11 = 4. * (x ** 2 + y - 11.) + 8. * x ** 2 + 2.
        h12 = 4. * x + 4. * y
        h21 = 4. * x + 4. * y
        h22 = 4. * (x + y ** 2 - 7.) + 8. * y ** 2 + 2.
        hessian = np.array([[h11, h12], [h21, h22]])
        return hessian

class Rosenbrock_Func():
    @staticmethod
    def eval(point):
        x, y = point[0], point[1]
        func = None
        ##############
        ##############
        func = (y - x**2)**2 + (1 - x)**2 / 100
        ##############
        ##############
        return func

    @staticmethod
    def gradient(point):
        x, y = point[0], point[1]
        grad = None
        ##############
        ##############
        dx = 2*(y - x**2)*(-2*x) + 2*(1 - x)*(-1) / 100
        dy = 2*(y - x**2)
        
        grad = np.array([dx, dy])
        ##############
        ##############
        return grad
    
    @staticmethod
    def hessian(point):
        x, y = point[0], point[1]
        h11 = 12 * x ** 2 - 4 * y + 1 / 50
        h12 = -4 * x
        h21 = -4 * x
        h22 = 2
        hessian = np.array([[h11, h12], [h21, h22]])
        return hessian


class Ackley_Func():
    @staticmethod
    def eval(point):
        func = None
        ##############
        ##############
        x, y = point[0], point[1]
        func = -20*np.exp(-0.2*np.sqrt(0.5*(x**2+y**2))) - np.exp(0.5*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))) + np.exp(1) + 20

        ##############
        ##############
        return func

    @staticmethod
    def gradient(point):
        grad = None
        ##############
        ##############
        x, y = point[0], point[1]

        dx = (2*x / np.sqrt(0.5*(x**2 + y**2))) * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) \
        + np.pi * np.sin(2*np.pi*x) * np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

        dy = (2*y / np.sqrt(0.5*(x**2 + y**2))) * np.exp(-0.2*np.sqrt(0.5*(x**2 + y**2))) \
        + np.pi * np.sin(2*np.pi*y) * np.exp(0.5*(np.cos(2*np.pi*x) + np.cos(2*np.pi*y)))

        grad = np.array([dx, dy])
        
        ##############
        ##############
        return grad
    
    @staticmethod
    def hessian(point):
        x, y = point[0], point[1]
        
        Hxx = (
            -2 * x**2 * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2)))
              / (5 * (x**2 + y**2))
            - 2 * np.sqrt(2) * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2))) / np.sqrt(x**2 + y**2)
            + 2 * np.sqrt(2) * x**2 * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2))) / (x**2 + y**2)**(1.5)
            - (np.pi**2)
              * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2)
              * np.sin(2 * np.pi * x)**2
            + 2 * (np.pi**2)
              * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2)
              * np.cos(2 * np.pi * x)
            + 2 * np.sqrt(2)
              * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2))) / np.sqrt(x**2 + y**2)
              * np.cos(2 * np.pi * x)
        )

        Hyy = (
            -2 * y**2 * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2)))
              / (5 * (x**2 + y**2))
            - 2 * np.sqrt(2) * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2))) / np.sqrt(x**2 + y**2)
            + 2 * np.sqrt(2) * y**2 * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2))) / (x**2 + y**2)**(1.5)
            - (np.pi**2)
              * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2)
              * np.sin(2 * np.pi * y)**2
            + 2 * (np.pi**2)
              * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2)
              * np.cos(2 * np.pi * y)
            + 2 * np.sqrt(2)
              * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2))) / np.sqrt(x**2 + y**2)
              * np.cos(2 * np.pi * y)
        )

        Hxy = (
            - (2 * x * y * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2)))) / (5 * (x**2 + y**2))
            + 2 * np.sqrt(2) * x * y * np.exp(-np.sqrt(x**2 + y**2) / (5 * np.sqrt(2))) / (x**2 + y**2)**(1.5)
            + (np.pi**2)
              * np.exp((np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)) / 2)
              * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)
        )

        return np.array([[Hxx, Hxy], [Hxy, Hyy]])