from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import normalize_dims

data_x = np.array([0.1, 0.9, 1.9, 2.3, 3, 4.1, 5.2, 5.9, 6.8, 8.1, 8.7, 9.2, 10.1,12])
data_y = np.array([20, 24, 27, 29, 32, 37.3, 36.4, 32.4, 28.5, 30, 38, 43, 40, 32])
data_x = normalize_dims(data_x)
data_y = normalize_dims(data_y)

class CrossValidationSplit:
    def __init__(self, x, y, n_split:int = 5, seed = 123) -> None:
        self.x = x
        self.y = y
        self.n_split = n_split
        self.iter = 0

        random.seed(seed)
        self.indices = list(range(len(x)))
        random.shuffle(self.indices)
        
        start = 0
        step = [len(x) // n_split] * n_split
        remain = len(x) % n_split
        for i in range(remain):
            step[i] += 1

        self.split_indices = []
        for i in range(n_split):
            stop = start + step[i]
            if i == -1:
                stop = len(start)
            self.split_indices.append(self.indices[start: stop])
            start = stop

    def __iter__(self):
        return self
    
    def __next__(self):
        if self.iter == self.n_split:
            raise StopIteration()
        
        x_train, y_train = [], []
        for i, indices in enumerate(self.split_indices):
            x = self.x[indices].copy()
            y = self.y[indices].copy()
            if i == self.iter:
                x_test = x
                y_test = y
            else:
                x_train.extend(x)
                y_train.extend(y)
        x_train = np.array(x_train)
        y_train = np.array(y_train)
        self.iter += 1

        x_train = normalize_dims(x_train)
        y_train = normalize_dims(y_train)
        x_test = normalize_dims(x_test)
        y_test = normalize_dims(y_test)

        return x_train, y_train, x_test, y_test

class PolynomialRegression:
    def __init__(self, degree:int) -> None:
        self.degree = degree

    def fit(self, x, y):
        poly = PolynomialFeatures(degree = self.degree)
        X_poly = poly.fit_transform(x)

        reg = LinearRegression()
        reg.fit(X_poly, y)
        
        coef = reg.coef_[0][1:].tolist()
        coef = np.flip(coef, axis=0)
        self.weight = [*coef, reg.intercept_[0]]

    def transform(self, x):
        def poly_line(x):
            f = 0
            for i, w in enumerate(self.weight):
                f += w * (x ** (self.degree - i))
            return f
        y = np.apply_along_axis(poly_line, axis = 0, arr = x)
        return y

def mse_loss(pred_y, y):
    return np.mean((pred_y - y) ** 2)

if __name__ == '__main__':
    val_loss_of_degree = []
    max_degree = 8
    n_split = 3
    num_figure = 2
    degrees = list(range(1, max_degree + 1))
    cur_fig = 0
    for degree_i, degree in enumerate(degrees):
        if degree_i % (max_degree // num_figure) == 0:
            cur_fig += 1
            plt.figure(figsize=(20, 20))

        degree_i %= (max_degree // num_figure)
        regressor = PolynomialRegression(degree)
        losses = []
        for split_i, (x_train, y_train, x_test, y_test) in enumerate(CrossValidationSplit(data_x, data_y, n_split=n_split)):
            regressor.fit(x_train, y_train)
            pred_y_test = regressor.transform(x_test)
            pred_y_all = regressor.transform(normalize_dims(data_x))
            loss = mse_loss(pred_y_test, y_test)
            losses.append(loss)

            ax = plt.subplot(max_degree // num_figure, n_split, (split_i + 1) + degree_i * n_split)
            ax.scatter(x_train, y_train, c='b', label='train_split')
            ax.scatter(x_test, y_test, c='r', label='test_split')
            ax.plot(data_x, pred_y_all)
            for x, pred_y, gt_y in zip(x_test, pred_y_test, y_test):
                ax.plot([x, x], [pred_y, gt_y], '--', color='k')
            ax.legend()
            ax.set_title("Degree = {}, test_split = {}, val_loss={:.2f}".format(degree, split_i + 1, loss))
        plt.tight_layout(h_pad=3)

        if degree_i == (max_degree // num_figure) - 1:
            plt.savefig('Q3-{}.png'.format(cur_fig))

        avg_loss = np.mean(losses)
        val_loss_of_degree.append(avg_loss)
    
    for degree, val_loss in enumerate(val_loss_of_degree):
        print("Degree = {}, avg_val_loss = {:10.2f}".format(degree + 1, val_loss))    
    print('Best Degree = {}'.format(np.argmin(val_loss_of_degree) + 1))
    plt.show()    
    