import numpy as np
from numpy.typing import NDArray

def build_poly_lines(w: NDArray[np.float64], dim: int) -> str:
    result = ''
    for i in range(dim + 1):
        if i != 0:
            result += ' + '
        if i == dim:
            result += '{:.4f}'.format(w[dim - i])
        else:
            result += '{:.4f}x^{}'.format(w[dim - i], dim - i)
    return result

def normalize_dims(x: np.ndarray):
    if len(x.shape) == 1:
        x = np.expand_dims(x, axis = 1)
    return x

def bracket(f, x1, h = 0.1): # from the book: Numerical methods in Engineering with Python
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

def gold_section(f, intv, h=1e-7):
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
         
    return xopt, f(xopt), n_iter

def compute_gradient(f, x, h = 1e-5) -> np.array: # x is a vector; the output is the gradient vector
    df = []
    for i in range(len(x)):
        xplus, xminus = x.copy(), x.copy()
        xplus[i] += h
        xminus[i] -= h
        df.append((f(xplus) - f(xminus)) / (2 * h))
    return np.array(df)