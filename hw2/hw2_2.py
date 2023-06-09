import numpy as np #import identity,array,dot,zeros,argmax
from math import sqrt
from typing import Tuple
import itertools
import matplotlib.pyplot as plt
from collections import defaultdict


def bracket(f, x1, h):#from the book: Numerical methods in Engineering with Python
    c = 1.618033989
    f1 = f(x1)
    x2 = x1 + h
    f2 = f(x2)
    # Determine downhill direction and change sign of h if needed
    if f2 > f1:
        h = -h
        x2 = x1 + h
        f2 = f(x2)
        # Check if minimum between x1 - h and x1 + h
        if f2 > f1: 
            return x2, x1 - h
    # Search loop
    for i in range (100):
        h = c * h
        x3 = x2 + h
        f3 = f(x3)
        if f3 > f2: 
            return x1,x3
        x1 = x2
        x2 = x3
        f1 = f2
        f2 = f3
    print("The bracket did not include a minimum")

def gold_section(f, intv, h=[]):
    if h == []:
        h = 1.0e-4
    alf = (5**0.5 -1)/2
    a = intv[0]
    b = intv[1]
    lam = a + (1 - alf) * (b-a); mu=a+alf*(b-a)
    fl=f(lam); fm=f(mu) #At first, 2 function evaluations are needed
    n_iter =1
    while (b-a)>h:
        n_iter = n_iter+1      
        if (fl>fm):
            a=lam
            lam=mu
            fl=fm
            mu=a+alf*(b-a)  
            fm=f(mu)   #In the while loop, only 1 more function evalution is needed
        else:
            b=mu
            mu=lam
            fm=fl
            lam=a+(1-alf)*(b-a)
            fl=f(lam)

    if f(a)<f(b): #compare f(a) with f(b), and xopt is the one with smaller func value
        xopt=a
    else:
        xopt=b    
         
    return xopt,f(xopt),n_iter


def powell(obj_fun, 
           x, 
           h: float = 0.1, 
           tol: float = 1e-5, 
           maxiter: int = 100):
    """ Powell's conjugate directions method
    """
    def f(s): 
        return obj_fun(x + s * v)    # F in direction of v

    n = len(x)                     # Number of design variables
    df = np.zeros(n)                  # Decreases of F stored here
    u = np.identity(n)                # Vectors v stored here by rows

    walk_points = [x]

    for j in range(maxiter):            # Allow for maximum cycles:
        x_old = x.copy()            # Save starting point
        f_old = obj_fun(x_old)
      # First n line searches record decreases of F
        for i in range(n):
            v = u[i]
            a, b = bracket(f, 0.0, h)
            s, fMin, ni = gold_section(f, [a, b])#search(f,a,b)
            df[i] = f_old - fMin
            f_old = fMin
            x = x + s * v
            walk_points.append(x)
      # Last line search in the cycle    
        v = x - x_old
        a, b = bracket(f, 0.0, h)
        s, fLast, ni = gold_section(f, [a,b]) #search(f,a,b)
        x = x + s * v
        walk_points.append(x)
      # Check for convergence
        if sqrt(np.dot(x - x_old, x - x_old) / n) < tol:
            walk_points = np.array(walk_points)
            return x, j + 1, walk_points
      # Identify biggest decrease & update search directions
        iMax = np.argmax(df)
        for i in range(iMax, n-1):
            u[i] = u[i+1]
        u[n-1] = v
    return None


def down_hill(obj_fun, 
                start_x: Tuple[float, float],
                tol = 1e-4,
                maxiter: int = 300):
    if isinstance(start_x, tuple):
        start_x = np.array(start_x)

    alpha, gamma, beta = 1.0, 2.0, 0.5
    num_var = len(start_x) # n
    # Generate n + 1 points
    points = np.zeros((num_var + 1, num_var)) # (n + 1, n)
    points[0] = start_x

    for i in range(1, num_var + 1):
        points[i] = points[0].copy()
        points[i, i - 1] += 4

    walk_points = [start_x]
    n_iter = 0

    while n_iter <= maxiter:
        n_iter += 1
        f_values = np.apply_along_axis(obj_fun, arr = points, axis=1)
        # Sort by obj function
        points = points[np.argsort(f_values)]
        best_point = points[0]
        worse_point = points[-1]
        f_values = np.sort(f_values)
        best_f_value = f_values[0]
        worse_f_value = f_values[-1]
    
        if np.sqrt(np.sum((best_point - worse_point) ** 2)) < tol:
            walk_points = np.array(walk_points) # (n, 2)
            return points[0] , n_iter, walk_points
        # Calculate the center of best N point
        center = np.mean(points[:-1, ], axis=0)
        walk_points.append(center)

        # Calculate reflection_point
        reflection = center + alpha * (center - worse_point)
        f_reflection = obj_fun(reflection)
        
        if best_f_value > f_reflection:
            expansion = center + gamma * (reflection - center)
            f_expansion = obj_fun(expansion)
            if f_reflection > f_expansion:
                points[-1] = expansion
            else:
                points[-1] = reflection
        else:
            second_worse_f_value = f_values[-2]
            if(second_worse_f_value >= f_reflection):
                points[-1] = reflection
            else:
                if f_reflection > worse_f_value:
                    x_p = worse_point
                    f_p = worse_f_value
                else:
                    x_p = reflection
                    f_p = f_reflection

                contraction = center + beta * (x_p - center)
                f_contraction = obj_fun(contraction)
                if f_contraction > f_p:
                    best_point = points[0].copy()
                    for j in range(num_var + 1):
                        points[j] = points[j] + (best_point - points[j]) / 2
                else:
                    points[-1] = contraction

    return None


def q2_obj_fun(x: Tuple[float, float]): 
    x1, x2 = x
    f = (-13 + x1 + ((5 - x2) * x2 - 2) * x2) ** 2 + \
        (-29 + x1 + ((x2 + 1) * x2 - 14) * x2) ** 2
    return f

def gen_intv(start: float, stop: float, n_split:int = 5):
    step = (stop - start) / (n_split - 1) 
    return np.arange(start, stop + step, step)

def draw_fig(min_start, method_name: str):
    for i, key in enumerate(min_start.keys()):
        x_start, n_iter, walk_points = min_start[key][0]
        
        plt.figure(figsize=(15, 15))
        plt.title("With {}, after {:3d} iterations".format(method_name, n_iter))
        plt.plot(walk_points[:, 0], walk_points[:, 1], marker='x', markeredgecolor='red')

        start_x1, start_x2 = walk_points[0]
        end_x1, end_x2 = walk_points[-1]
        info = '  Start\n  x=({:.1f},{:.1f}), y={:.2f}'.format(start_x1, start_x2, q2_obj_fun((start_x1, start_x2)))
        plt.text(start_x1, start_x2, s=info, c='g')
        info = '  Minimum\n  x=({:.1f},{:.1f}), y={:.2f}'.format(end_x1, end_x2, q2_obj_fun((end_x1, end_x2)))
        plt.text(end_x1, end_x2, s=info, c='r')

        plt.xlabel("x1")
        plt.ylabel("x2")
        plt.show()

def find_minimum(method, intvs, method_name: str):
    min_start = defaultdict(list)
    for x_start in list(itertools.permutations(intvs, 2)):
        result = method(q2_obj_fun, np.array(x_start))
        
        if result == None:
            continue
        x_opt, n_iter, walk_points = result

        x1, x2 = x_opt
        x1 = round(x1, 1)
        x2 = round(x2, 1)
        min_start[(x1, x2)].append([x_start, n_iter, walk_points])

    print("With {}:".format(method_name))
    for key, datas in min_start.items():
        print('Optimum Point = {}'.format(key))
        print("Start at:")
        for data in datas:
            print(data[0], end=' ')
        print()
    draw_fig(min_start, method_name)
    print()

if __name__ == '__main__':
    intvs = gen_intv(-5, 15, 5)

    find_minimum(powell, intvs, 'powell')
    find_minimum(down_hill, intvs, 'down_hill')