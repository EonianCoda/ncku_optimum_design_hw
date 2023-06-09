import math
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import defaultdict


def golden_section_method(obj_fun, 
                        intv: Tuple[float, float], 
                        h: float = 1e-4,
                        find_min: bool = True):
    alpha = (5 ** 0.5 - 1) / 2
    a = float(intv[0])
    b = float(intv[1])
    left = a + (1 - alpha) * (b - a)
    right = a + alpha * (b - a)
    f_left = obj_fun(left)
    f_right = obj_fun(right) #At first, 2 function evaluations are needed
    n_iter = 1

    trajectories = [f_left]
    while (b - a) > h:
        n_iter = n_iter + 1

        if (find_min and (f_left > f_right)) or ((not find_min) and (f_left < f_right)):
            a = left
            left = right
            f_left = f_right
            right = a + alpha * (b - a)  
            f_right = obj_fun(right) #In the while loop, only 1 more function evalution is needed
        else:
            b = right
            right = left
            f_right = f_left
            left = a + (1 - alpha) * (b - a)
            f_left = obj_fun(left)

        trajectories.append(f_left)
    # compare f(a) with f(b), and xopt is the one with smaller func value
    if find_min:
        x_best = a if obj_fun(a) < obj_fun(b) else b
    else:
        x_best = a if obj_fun(a) > obj_fun(b) else b

    return x_best, obj_fun(x_best), n_iter, trajectories

class FibonacciFun:
    def __init__(self):
        self.values = [1, 1]

    def __call__(self, n: int) -> int:
        last_n = len(self.values)
        if n >= last_n:
            for i in range(last_n, n + 1):
                new_v = self.values[i - 2] + self.values[i - 1]
                self.values.append(new_v)
        return self.values[n]

    def find_n(self, min_F_value: float):
        n = 4
        while self.__call__(n) < min_F_value:
            n += 1
        return n

def fibonacci_method(obj_fun, 
                     intv: Tuple[float, float], 
                     h: float = 1e-4,
                     eps: float = 1e-4,
                     find_min: bool = True):
    a, b = intv
    min_F = (b - a) / h
    fibonacci_fun =  FibonacciFun()
    n = fibonacci_fun.find_n(min_F) 

    left = a + ((fibonacci_fun(n - 2) / fibonacci_fun(n)) * (b - a))
    right = a + ((fibonacci_fun(n - 1) / fibonacci_fun(n)) * (b - a))

    f_left = obj_fun(left)
    f_right = obj_fun(right)

    n_iter = 1
    k = 1

    trajectories = [f_left]
    while k != n - 1:
        if (find_min and (f_left > f_right)) or ((not find_min) and (f_left < f_right)):
            a = left
            left = right
            f_left = f_right
            right = a + ((fibonacci_fun(n - k - 1) / fibonacci_fun(n - k)) * (b - a))
            f_right = obj_fun(right)
        else:
            b = right
            right = left
            f_right = f_left
            left = a + ((fibonacci_fun(n - k - 2) / fibonacci_fun(n - k)) * (b - a))
            f_left = obj_fun(left)
        n_iter += 1
        k += 1
        trajectories.append(f_left)
    
    right = left + eps
    f_right = obj_fun(right)
    if (find_min and (f_left > f_right)) or ((not find_min) and (f_left < f_right)):
        a = left
    else:
        b = right

    xopt = a if obj_fun(a) < obj_fun(b) else b
    return xopt, obj_fun(xopt), n_iter, trajectories

def q1_obj_fun(x: float):
    return (x ** 2) * math.cos(x)

def gen_intv(start: float, stop: float, n_split:int = 5):
    step = (stop - start) / (n_split - 1) 
    return np.round(np.arange(start, stop + step, step), 4)

def draw_figure(optimum_trajectories: dict, optimum_points: dict, optimum_n_iters: dict, method_name: str):
    plt.figure(figsize=(15, 10), tight_layout=True)
    plt.suptitle(method_name)
    num_min = len(optimum_trajectories['min'])
    num_max = len(optimum_trajectories['max'])
    
    for key_i, key in enumerate(['min', 'max']):
        for i, (trajectories, opt_point, opt_n_iter) in enumerate(zip(optimum_trajectories[key], optimum_points[key], optimum_n_iters[key])):
            ax = plt.subplot(2, max(num_min, num_max), i + 1 + max(num_min, num_max) * key_i)
            ax.scatter(list(range(1, len(trajectories) + 1)), trajectories, marker='o')
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_xlabel('Iteration Number')
            ax.set_ylabel('Function Value')
            ax.set_title("Local {}: x = {:+.4f}, f(x) = {:+.3f}, n_iter={:2d}".format(key, opt_point, trajectories[-1], opt_n_iter))
    plt.show()


def find_optimum(method, intvs):
    optimum_trajectories = defaultdict(list)
    optimum_points = defaultdict(list)
    optimum_n_iters = defaultdict(list)

    method_name = method.__name__
    for i in range(len(intvs) - 1):
        intv = [intvs[i], intvs[i + 1]]
        print("Start with interval [{:.4f}, {:.4f}]".format(intv[0], intv[1]))
        
        # Find Minimum
        for find_min in [True, False]:    
            key = 'min' if find_min else 'max'

            x_best, f_best, n_iter, trajectories = method(q1_obj_fun, intv, find_min=find_min)
            if abs(x_best - intv[0]) > 1e-3 and abs(x_best - intv[1]) > 1e-3:
                print('With {}: after {:2d} iteratoins, the local {} point is {:+.4f} with func value {:+.3f}'.format(method_name,
                                                                                                                    n_iter, key, 
                                                                                                                    x_best, 
                                                                                                                    f_best))
                optimum_trajectories[key].append(trajectories)
                optimum_points[key].append(x_best)
                optimum_n_iters[key].append(n_iter)
    # Draw figure  
    draw_figure(optimum_trajectories, optimum_points, optimum_n_iters, method_name)


if __name__ == '__main__':
    intvs = gen_intv(-2, 2, 6)
    print(intvs)
    find_optimum(golden_section_method, intvs)
    find_optimum(fibonacci_method, intvs)