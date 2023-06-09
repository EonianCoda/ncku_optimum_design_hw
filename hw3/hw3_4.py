import numpy as np
from numpy.typing import NDArray
from numpy import linalg as LA
from typing import Any, Tuple
import matplotlib.pyplot as plt
import math
from utils import normalize_dims, bracket, gold_section, compute_gradient, build_poly_lines

data_x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
data_y = np.array([20, 24, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 38, 43, 40, 32])

poly_line_dims = 3

def poly_line(w, x, max_dim):
    f = 0
    for i in range(max_dim):
        f += w[i] * (x ** i)
    return f

class MyFinalCreativeFun:
    @staticmethod
    def num_weight():
        return poly_line_dims + 4
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        pred_y +=  w[poly_line_dims] * math.sin(w[poly_line_dims + 1] * x + w[poly_line_dims + 2]) + \
                    w[poly_line_dims + 3] * math.sin(x ** 2)
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2 + {} sin({}x + {}) + {}sin(x^2)'.format(*'abcdefg')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f} sin({:.3f}x + {:.3f}) + {:.3f}sin(x^2)'.format(*w)

class MyCreativeFun1:
    @staticmethod
    def num_weight():
        return 2
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y =  w[0] + w[1] * math.sin(x)
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {} sin(x)'.format(*'abcde')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f} sin(x)'.format(*w)

class MyCreativeFun2:
    @staticmethod
    def num_weight():
        return poly_line_dims + 1
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        pred_y +=  w[poly_line_dims] * math.sin(x)
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2 + {} sin(x)'.format(*'abcd')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f} sin(x)'.format(*w)
    
class MyCreativeFun3:
    @staticmethod
    def num_weight():
        return poly_line_dims + 3
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        pred_y +=  w[poly_line_dims] * math.sin(w[poly_line_dims + 1] * x)
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2 + {} sin({}x)'.format(*'abcde')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f} sin({:.3f}x)'.format(*w)
    
class MyCreativeFun4:
    @staticmethod
    def num_weight():
        return poly_line_dims + 4
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        pred_y +=  w[poly_line_dims] * math.sin(w[poly_line_dims + 1] * x) + w[poly_line_dims + 2] * math.cos(w[poly_line_dims + 3] * x)
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2 + {} sin({}x) + {} cos({}x)'.format(*'abcdefg')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f} sin({:.3f}x) + {:.3f} cos({:.3f}x)'.format(*w)

class MyCreativeFun5:
    @staticmethod
    def num_weight():
        return poly_line_dims + 6
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        pred_y +=  w[poly_line_dims] * math.sin(w[poly_line_dims + 1] * x) \
        + w[poly_line_dims + 2] * math.cos(w[poly_line_dims + 3] * x) \
        + w[poly_line_dims + 4] * math.sin(w[poly_line_dims + 5] * x)
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2 + {} sin({}x) + {} cos({}x) + {} sin({}x)'.format(*'abcdefghi')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f} sin({:.3f}x) + {:.3f} cos({:.3f}x) + {:.3f} sin({:.3f}x)'.format(*w)

class MyCreativeFun6:
    @staticmethod
    def num_weight():
        return poly_line_dims + 8
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        pred_y +=  w[poly_line_dims] * math.sin(w[poly_line_dims + 1] * x) \
        + w[poly_line_dims + 2] * math.cos(w[poly_line_dims + 3] * x) \
        + w[poly_line_dims + 4] * math.sin(w[poly_line_dims + 5] * x) \
        + w[poly_line_dims + 6] * math.cos(w[poly_line_dims + 7] * x) 
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2 + {} sin({}x) + {} cos({}x) + {} sin({}x) + {} cos({}x)'.format(*'abcdefghijk')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f} sin({:.3f}x) + {:.3f} cos({:.3f}x) + {:.3f} sin({:.3f}x) + {:.3f} cos({:.3f}x)'.format(*w)

class MyCreativeFun7:
    @staticmethod
    def num_weight():
        return poly_line_dims + 10
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        pred_y +=  w[poly_line_dims] * math.sin(w[poly_line_dims + 1] * x) \
        + w[poly_line_dims + 2] * math.cos(w[poly_line_dims + 3] * x) \
        + w[poly_line_dims + 4] * math.sin(w[poly_line_dims + 5] * x) \
        + w[poly_line_dims + 6] * math.cos(w[poly_line_dims + 7] * x) \
        + w[poly_line_dims + 8] * math.sin(w[poly_line_dims + 9] * x)
        return pred_y
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2 + {} sin({}x) + {} cos({}x) + {} sin({}x) + {} cos({}x) + {} sin({}x)'.format(*'abcdefghijkln')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2 + {:.3f} sin({:.3f}x) + {:.3f} cos({:.3f}x) + {:.3f} sin({:.3f}x) + {:.3f} cos({:.3f}x) + {:.3f} sin({:.3f}x)'.format(*w)

class MyQuadraticFun:
    @staticmethod
    def num_weight():
        return poly_line_dims
    
    def __call__(self, w: np.ndarray, x: float) -> float:
        global poly_line_dims
        w = np.squeeze(w)
        pred_y = poly_line(w, x, poly_line_dims)
        return pred_y
    
    def get_string(self) -> str:
        return 'f(x)= {} + {}x + {}x^2'.format(*'abc')
    def get_string_with_weight(self, w) -> str:
        return 'f(x)= {:.3f} + {:.3f}x + {:.3f}x^2'.format(*w)

def my_fun_wrapper(fun, w: np.ndarray) -> float:
    return lambda x: fun(w, x)

def obj_fun_wrapper(fun):
    def obj_fun(w: NDArray[np.float64]):
        ff = 0.0
        for data_i in range(len(data_x)):
            x = data_x[data_i]
            y = data_y[data_i]
            pred_y = fun(w, x)
            ff += (pred_y - y) ** 2
        return ff
    return obj_fun

def DFP(obj_fun, 
        start_x: NDArray[np.float64], 
        h = 1e-4,
        n = 40) -> Tuple[NDArray[np.float64], int]:
    
    def obj_fun_of_search(x, search_direction):
        def fun(step):
            return obj_fun(x + step * search_direction)
        return fun
    
    x = start_x
    b = np.identity(len(x))
    iter_i = 1
    while iter_i <= 10000:
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
    print('Not converge')
    return x, iter_i 
if __name__ == '__main__':

    methods = [MyQuadraticFun, MyCreativeFun1, MyCreativeFun2, MyCreativeFun3, MyCreativeFun4, MyCreativeFun5, MyCreativeFun6]

    plt.figure(figsize=(16, 16))
    row = 4
    for method_i, method in enumerate(methods):
        ax = plt.subplot(row, 2, method_i + 1)
        method = method()
        start_x = np.array([0.2] * (method.num_weight()), dtype = np.float64)
        
        if len(start_x) >= 11:
            # cos
            start_x[-1] = -0.2
            start_x[-2] = -0.2
            # sin
            start_x[-3] = -0.2
            start_x[-4] = -0.2
        elif len(start_x) >= 9:
            # sin
            start_x[-1] = -0.2
            start_x[-2] = -0.2
            
        obj_fun = obj_fun_wrapper(method)
        best_w, iter_k = DFP(obj_fun, start_x)
        pred_y = np.apply_along_axis(my_fun_wrapper(method, best_w), axis=1, arr=np.expand_dims(data_x, 1))
        error = np.mean((normalize_dims(pred_y) - normalize_dims(data_y)) ** 2)

        for x, y1, y2 in zip(data_x, data_y, pred_y):
            ax.plot([x, x], [y1, y2], '--', c='k', linewidth=1.0)
        fun_content = method.get_string()
        print('With function {}, iter {} times:\n=> {}'.format(fun_content,
                                                                     iter_k,
                                                                     method.get_string_with_weight(best_w)))
        print('Avg MSE Loss = {:.3f}\n'.format(error))

    
        ax.scatter(data_x, data_y)

        line_x = np.arange(0, 13, 0.01)
        line_y = np.apply_along_axis(my_fun_wrapper(method, best_w), axis=1, arr=np.expand_dims(line_x, 1)) 
        
        ax.plot(line_x, line_y, c='k', linewidth=0.25, label='real line')
        ax.plot(data_x, pred_y, c='r', marker='x', label='fit data line')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_title('{}.{}, Loss = {:.3f}'.format(method_i + 1, fun_content, error))
        ax.legend()
    plt.tight_layout(h_pad=2)
    plt.savefig('Q4.png')
    plt.show()