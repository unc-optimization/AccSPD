"""
This script compare our Algorithm 2 with SPDHG algorithms in the following strongly convex example:

    Soft margin SVM without bias
    min_{x, x0} sum_{j=1,...,m} f[j](b_j * (A[j,:] * x) ) + g(x),
    where f[j](r) = max(0 , 1 - r), g(x) = beta/2 norm(x)^2.

with the following data:

    "w8a", "rcv1", "real-sim", and "news20" from the LIBSVM datasets website.

If you want to test more SVM data:

    add your data under  the "./data/" folder and change the "data_name" variable in this script.

This algorithms are implemented by using single coordinate.

References:

    [1] A. Chambolle, M. J. Ehrhardt, P. Richtárik, and C.-B. Schönlieb. Stochastic primal-dual hybrid gradient
    algorithm with arbitrary sampling and imaging applications. SIAM J. Optim., 28(4):2783–2808, 2018.

    [2] T. Goldstein, E. Esser, and R. Baraniuk. Adaptive primal-dual hybrid gradient methods for saddle point
    problems. Tech. Report., pages 1–26, 2013. http://arxiv.org/pdf/1305.0546v1.pdf.
"""

from __future__ import division, print_function

import numpy as np
import math
from scipy.sparse import linalg
import scipy.sparse as sp
import sys

from solver_single.original_alg2 import original_alg2
from solver_single.spdhg import spdhg
from sklearn.datasets import load_svmlight_file

# Import Data
# c, b, A, beta
data_name = "w8a"
X_train, y_train = load_svmlight_file("data/" + data_name)
m = y_train.size
A = sp.diags(y_train).dot(X_train)
At = A.transpose()
c = np.ones(m)
beta = 0.0001 * m
m, d = A.shape


# Define objective functions
def primal_obj_func(x):
    r = A.dot(x)
    return np.maximum(0, 1 - r).dot(c) + (beta / 2) * x.dot(x)


def dual_obj_func(y):
    if (y >= -c).all() and (y <= 0).all():
        z = -At.dot(y)
        return - y.sum() - (1 / 2 / beta) * z.dot(z)
    else:
        return - math.inf


# fix a seed
np.random.seed(0)

# set batch size
batch_size = 1


def fun_select_x(k):
    return np.random.choice(d, batch_size).tolist()


def fun_select_r(k):
    return np.random.choice(m, batch_size).tolist()


# set number of epoch
num_epoch = 3

# fix the initial point and iterations
x0 = np.zeros(d)
y0 = np.zeros(m)
primal_obj_x0 = primal_obj_func(x0)
dual_obj_y0 = dual_obj_func(y0)
gap0 = primal_obj_x0 - dual_obj_y0

# decide to run which algorithms
alg_list = dict(original_alg2=1, adapted_alg2=0, adapted_alg2_V2=0, adapted_alg2_full=0, spdhg=1, pdhg=1, spdc=0,
                sdca=0)

# Define the solution set
Sol = dict()

"""
Original Algorithm 2
"""
if alg_list["original_alg2"]:
    # Define proximal operators
    def proximal_x(j, x, sigma):
        return x / (beta * sigma + 1)


    def proximal_r(r, sigma):
        return np.minimum(np.maximum(r, 1), r + sigma)


    def proximal_r_star(r, sigma):
        return np.minimum(np.maximum(r - sigma, -1), 0)


    if_average = 1
    tau0 = batch_size / d  # 1/(blk_x*blk_r)
    rho0 = 10 / linalg.norm(A)  # penalty parameter for r , 8 for w8a
    sigma = 10 / linalg.norm(A)  # penalty parameter for x, 8 for w8a
    c_eta = 1  # 1 for w8a
    c_tau = 2 / tau0  # 2 for w8a
    if data_name == "covtype":
        rho0 = 0.5 / linalg.norm(A)  # penalty parameter for r
        sigma = 15 / linalg.norm(A)  # penalty parameter for x
        c_eta = 1
        c_tau = 100 / tau0

    Sol["original_alg2"] = original_alg2(x0, proximal_x, proximal_r_star, A, if_average,
                                         int(num_epoch * d / batch_size),
                                         primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                         fun_select_x=fun_select_x,
                                         rho0=rho0, tau0=tau0, sigma=sigma, c_tau=c_tau, c_eta=c_eta, verbose=1,
                                         print_dist=int(num_epoch * d / batch_size / 50))
    Sol["original_alg2"]["gap"] = [gap0] + Sol["original_alg2"]["gap"]

"""
SPDHG algorithm
"""
if alg_list["spdhg"]:
    # Define proximal operators
    def proximal_r(i, r, sigma):
        return np.minimum(np.maximum(r, 1), r + sigma)


    def proximal_r_star(i, r, sigma):
        return np.minimum(np.maximum(r - sigma, -1), 0)


    def proximal_x(x, tau):
        return x / (beta * tau + 1)


    tau = 10 / linalg.norm(A)
    sigma = 10 / linalg.norm(A)
    if data_name == "leu":
        tau = 1 / linalg.norm(A)
        sigma = 1 / linalg.norm(A)

    if_average = 1
    Sol["spdhg"] = spdhg(x0, proximal_x, proximal_r_star, A, tau, sigma, if_average, int(num_epoch * m / batch_size),
                         primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func, verbose=1,
                         fun_select_r=fun_select_r,
                         print_dist=int(num_epoch * m / batch_size / 50))
    Sol["spdhg"]["gap"] = [gap0] + Sol["spdhg"]["gap"]

"""
Plot the result
"""

import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rc

matplotlib.use('TkAgg')
rc('text', usetex=True)

# create the figure
fig_svm = plt.figure()
markersize = 8
mark_num = 10

total_iter = len(Sol["original_alg2"]["gap"])
iters = np.array(range(0, total_iter))
dist = total_iter / num_epoch
gap = np.array(Sol["original_alg2"]["gap"]) / m
plt.semilogy(iters / dist, gap, 'C1-', marker='o', markevery=int(total_iter / mark_num),
             markersize=markersize, label=r'Original Algorithm 2')

total_iter = len(Sol["spdhg"]["gap"])
iters = np.array(range(0, total_iter))
dist = total_iter / num_epoch
gap = np.array(Sol["spdhg"]["gap"]) / m
plt.semilogy(iters / dist, gap, 'C3-', marker='v', markevery=int(total_iter / mark_num),
             markersize=markersize, label=r'SPDHG')

plt.title("SVM Single Sample - " + data_name + ": m = " + str(m) + ", n = " + str(d))
plt.xlabel("Epochs")
plt.ylabel("Duality Gap")
plt.legend()
plt.show()
