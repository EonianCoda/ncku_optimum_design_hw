import math
import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA
from typing import Tuple
from utils import bracket, gold_section, compute_gradient

data_x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1, 12])

def obj_wrapper(K: float, alpha: float, r_k: float):
    def obj_fun(weights: NDArray[np.float64]):
        x, y = weights
        return K / (x * (y ** 2)) + r_k * (x**2 + y**2 - alpha**2) ** 2
    return obj_fun

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
    start_x = np.array([0.1, 0.1])
    
    K = 10  # meterial coef
    alpha = 10 # radius
    r_k = 1000 # penalty term
    f_w, iter_k =  DFP(obj_wrapper(K, alpha, r_k), start_x = start_x)
    
    print('With DFP Method, iter {:2d} times, x = {:.3f}, y = {:.3f}'.format(iter_k, f_w[0], f_w[1]))
    analytical_f_w = [alpha / math.sqrt(3), math.sqrt(2) * (alpha / math.sqrt(3))]
    print('Analytical Sol: x = {:.3f}, y = {:.3f}'.format(analytical_f_w[0], analytical_f_w[1]))