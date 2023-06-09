from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
from utils import normalize_dims

data_x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
data_y = np.array([20, 24, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 38, 43, 40, 32])
data_x = normalize_dims(data_x)
data_y = normalize_dims(data_y)

def linear_reg(x: np.ndarray, y: np.ndarray):
    reg = LinearRegression()
    reg.fit(x, y)
    return reg.coef_[0][0], reg.intercept_[0]

def quadratic_reg(x:list, y: list):
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(x)

    reg = LinearRegression()
    reg.fit(X_poly, y)
    
    coef = reg.coef_[0][1:]
    a, b = coef[1], coef[0]
    return a, b, reg.intercept_[0]

if __name__ == '__main__':
    a, b = linear_reg(data_x, data_y)
    linear_f = lambda x: a * x + b

    linear_x = np.arange(0, 12, 0.01)
    linear_line_y = np.apply_along_axis(linear_f, axis = 0, arr = linear_x)
    linear_equ = 'y= {:.4f}x+{:.4f}'.format(a, b)

    a, b, c = quadratic_reg(data_x, data_y)
    quadratic_f = lambda x: a * x * x + b * x + c 
    quadratic_line_y = np.apply_along_axis(quadratic_f, axis=0, arr=linear_x)
    quadratic_equ = 'y= {:.4f}x^2+{:.4f}x+{:.4f}'.format(a, b, c)

    print('linear: ', linear_equ)
    print('quadratic: ', quadratic_equ)
    plt.figure()
    plt.scatter(data_x, data_y)

    plt.plot(linear_x, linear_line_y, label='linear: {}'.format(linear_equ), c='b')
    plt.plot(linear_x, quadratic_line_y, label='quadratic: {}'.format(quadratic_equ), c='r')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    plt.savefig('Q1.png')
    plt.show()