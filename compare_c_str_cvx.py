"""
This script verifies that choosing different update rules for Algorithm 2 in the strongly convex case
will cause different convergence rate. We illustrate this point by using the following example:

    Soft margin SVM without bias
    min_{x, x0} sum_{j=1,...,m} f[j](b_j * (A[j,:] * x) ) + g(x),
    where f[j](r) = c_j * max(0 , 1 - r), g(x) = beta/2 norm(x)^2

with the following data:

    "a8a", "rcv1" from the LIBSVM data set website.

If you want to test more SVM data:

    add your data under  the "./data/" folder.
"""

from __future__ import division, print_function

import numpy as np
import math
from scipy import sparse as sp
from argParse import argParser_block
import time
from sklearn.datasets import load_svmlight_file
from solver_block.original_alg2_str_cvx import original_alg2_str_cvx
from solver_block.original_alg2_str_cvx1 import original_alg2_str_cvx1

# Read Parameters
num_blk, data_name, num_epoch = argParser_block()
if data_name is None:
    data_name = "a8a"

num_blk = int(num_blk)
num_epoch = int(num_epoch)

# Import Data
# c, b, A, beta
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

# fix the initial point
x0 = np.zeros(d)

# Define the solution set
Sol = dict()

# partition into blk_x blocks
blk_x = 32
bat_size_x = d // blk_x + 1
Ablk_x = []
index_x = []
for k in range(blk_x):
    mask = range((k % blk_x) * bat_size_x, min((k % blk_x + 1) * bat_size_x, d))
    index_x.append(mask)
    Ablk_x.append(A[:, mask])
Atblk_x = [Ablk_x[i].transpose() for i in range(blk_x)]

"""
Original Algorithm 2 In Strongly Convex Case
"""


# Define proximal operators
def proximal_x(x, sigma):
    return x / (beta * sigma + 1)


def proximal_r(r, sigma):
    return np.minimum(np.maximum(r, 1), r + sigma)


def proximal_r_star(r, sigma):
    return np.minimum(np.maximum(r - sigma, -1), 0)


# set number of iteration
niter = int(blk_x * num_epoch)

if_average = 1
tau0 = 1 / blk_x  # 1/(blk_x*blk_r)
c_eta = 2

# case 1: Theorem 4.2
rho0 = 1 / sp.linalg.norm(A)  # penalty parameter for r
sigma = 1 / sp.linalg.norm(A)  # penalty parameter for x
Sol["original_alg2_case1"] = original_alg2_str_cvx(x0, proximal_x, proximal_r_star, A, Ablk_x, Atblk_x, index_x,
                                                   if_average, niter,
                                                   primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                                   rho0=rho0, tau0=tau0, sigma=sigma, c_eta=c_eta, verbose=1,
                                                   print_dist=int(max(niter/50, 1)))
# case 2: Theorem 4.3
rho0 = 2 / sp.linalg.norm(A)  # penalty parameter for r
sigma = 2 / sp.linalg.norm(A)  # penalty parameter for x
c_tau = 3 / tau0
Sol["original_alg2_case2"] = original_alg2_str_cvx1(x0, proximal_x, proximal_r_star, A, Ablk_x, Atblk_x, index_x,
                                                    if_average, niter,
                                                    primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                                    rho0=rho0, tau0=tau0, sigma=sigma, c_tau=c_tau, c_eta=c_eta,
                                                    verbose=1,
                                                    print_dist=int(max(niter/50, 1)))

"""
save data
"""
import pickle

data = dict(dim_dual=m, dim_primal=d, blk_x=blk_x)

# calculate the initial gap
x0 = np.zeros(d)
y0 = np.zeros(m)
primal_obj_x0 = primal_obj_func(x0)
dual_obj_y0 = dual_obj_func(y0)
gap0 = primal_obj_x0 - dual_obj_y0

Sol["original_alg2_case1"]["gap"] = [gap0] + Sol["original_alg2_case1"]["gap"]
Sol["original_alg2_case2"]["gap"] = [gap0] + Sol["original_alg2_case2"]["gap"]

data["sol"] = Sol
pickle.dump(data, open("result/compare_c/svm_str_cvx_" + data_name, "wb"))

"""
Plot the result
"""
import pickle
import matplotlib.pyplot as plt
import math
import numpy as np
from matplotlib import rc

rc('text', usetex=True)

# load data
if 'data_name' not in locals():
    data_name = "a8a"
data = pickle.load(open("result/compare_c/svm_str_cvx_" + data_name, "rb"))
m = data["dim_dual"]
d = data["dim_primal"]
blk_x = data["blk_x"]

# create the figure
fig_compare_c = plt.figure(figsize=(20, 6))

inter_iter = blk_x
total_iter = len(data["sol"]["original_alg2_case1"]["gap"])
xticks = range(inter_iter, total_iter, inter_iter)
mask = np.array(range(inter_iter, total_iter))
num_epoch = len(xticks)

gap = np.array(data["sol"]["original_alg2_case1"]["gap"]) / m
num_points = len(mask)
num_mark = 20
markers_on = [int(math.exp(i * math.log(num_points) / num_mark)) for i in range(2, num_mark)]
plt.loglog(mask / inter_iter, gap[mask], 'C1-', marker='o', markevery=markers_on,
           markersize=10, label=r'Algorithm 2 (Parameters Updated by Theorem 4.2)')

inter_iter = blk_x
total_iter = len(data["sol"]["original_alg2_case2"]["gap"])
xticks = range(inter_iter, total_iter, inter_iter)
mask = np.array(range(inter_iter, total_iter))
num_epoch = len(xticks)

gap = np.array(data["sol"]["original_alg2_case2"]["gap"]) / m
num_points = len(mask)
num_mark = 20
markers_on = [int(math.exp(i * math.log(num_points) / num_mark)) for i in range(2, num_mark)]
plt.loglog(mask / inter_iter, gap[mask], 'b-', marker='s', markevery=markers_on,
           markersize=10, label=r'Algorithm 2 (Parameters Updated by Theorem 4.3)')

# # plot 1/k and 1/k^2
num_points = num_epoch
num_mark = 20
markers_on = [int(math.exp(i * math.log(num_points) / num_mark)) for i in range(2, num_mark)]
plt.loglog(range(1, num_epoch + 1), gap[inter_iter] / np.array(range(1, num_epoch + 1)) ** 2,
           'C3-', marker='*', markevery=markers_on, markersize=10, label=r'$\mathcal{O}(1/K^2)$')

plt.suptitle("Algorithm 2 in Strongly Convex- " + data_name + \
             ": m = " + str(m) + ", n = " + str(d))
plt.xlim(1, num_epoch + 1)
plt.xlabel("Epochs")
plt.ylabel("Duality Gap")
plt.legend(loc='lower left')
plt.show()
