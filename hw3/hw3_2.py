import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA
from typing import Tuple
from utils import bracket, gold_section, compute_gradient, build_poly_lines

data_x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1, 12])
data_y = np.array([20, 24, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 38, 43, 40, 32])

def poly_fun_wrapper(w: NDArray[np.float64]):
    def f(x):
        dim = len(w)
        v = 0
        for i in range(dim):
            v += w[i] * (x ** i)
        return v
    return f

def obj_fun(weights: NDArray[np.float64]):
    sum_f = 0.0
    poly_fun = poly_fun_wrapper(weights)
    for x, y in zip(data_x, data_y):
        sum_f += (poly_fun(x) - y) ** 2
    return sum_f

def conjugate_gradient(obj_fun, 
                       x: NDArray[np.float64], 
                       h = 1e-4) -> Tuple[NDArray[np.float64], int]:
    def obj_fun_of_search(x, search_direction):
        return lambda step: obj_fun(x + step * search_direction)
    iter_k = 1 
    n = len(x)
    while iter_k <= 1000:
        grad_last = compute_gradient(obj_fun, x)
        search_direction = -grad_last

        if LA.norm(search_direction, 2) < h:
            return x, iter_k
    
        iter_k += 1
        for _ in range(n):
            a, b = bracket(obj_fun_of_search(x, search_direction), 0.0, h = 0.1)
            step, fLast, ni = gold_section(obj_fun_of_search(x, search_direction), [a, b]) 
            x = x + step * search_direction
            grad = compute_gradient(obj_fun, x)
            beta = (LA.norm(grad, 2) ** 2) / (LA.norm(grad_last, 2) ** 2)

            search_direction = -grad + beta * search_direction
            grad_last = grad
    print("Not converge")
    return x, iter_k

def DFP(obj_fun, 
        start_x: NDArray[np.float64], 
        h = 1e-4) -> Tuple[NDArray[np.float64], int]:
    
    def obj_fun_of_search(x, search_direction):
        return lambda step: obj_fun(x + step * search_direction)
    
    x = start_x
    b = np.identity(len(x))
    iter_i = 1
    while True:
        if iter_i != 0 and iter_i % 5 == 0:
            b = np.identity(len(x))
        last_grad = compute_gradient(obj_fun, x)
        search_direction = - np.matmul(b, last_grad)

        intv_a, intv_b = bracket(obj_fun_of_search(x, search_direction), 0.0, h = 0.1)
        step, fLast, ni = gold_section(obj_fun_of_search(x, search_direction), [intv_a, intv_b]) 
        
        x = x + step * search_direction
        grad = compute_gradient(obj_fun, x)
        if LA.norm(grad, 2) < h:
            return x, iter_i
        
        search_direction = np.expand_dims(search_direction, axis=1)
        g = np.expand_dims(grad - last_grad, axis= 1)
        bg = np.matmul(b, g)
        m = step * np.matmul(search_direction, search_direction.T) / np.matmul(search_direction.T, g)
        n = - np.matmul(bg, bg.T) / np.matmul(np.matmul(g.T, b), g)
        b = b + m + n
        iter_i += 1

if __name__ == '__main__':
    plt.figure(figsize=(16,8))
    colors = ['b', 'r']
    methods = [conjugate_gradient, DFP]
    for method_i, method in enumerate(methods):
        method_name = method.__name__

        ax = plt.subplot(1, len(methods), method_i + 1)
        ax.scatter(data_x, data_y)
        for dim in range(1, 2 + 1):
            start_x = np.array([0.1] * (dim + 1), dtype = np.float64)
            f_w, iter_k = method(obj_fun, start_x)
            
            print('With {} method, iter {:2d} times, best {}-dim function = {}'.format(method.__name__, 
                                                                                        iter_k, 
                                                                                        dim, 
                                                                                        build_poly_lines(f_w, dim)))
            
            label = '{}-dim f(x) = {}'.format(dim, build_poly_lines(f_w, dim))
            linear_x = np.arange(0, 12, 0.01)
            ax.plot(linear_x, np.apply_along_axis(poly_fun_wrapper(f_w), axis=0, arr=linear_x), label=label, c=colors[dim - 1])
        ax.legend()
        ax.set_xlabel('x')
        ax.set_title("{} Method".format(method_name))
        ax.set_ylabel('y')
    plt.tight_layout()
    plt.savefig('Q2')
    plt.show()