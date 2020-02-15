"""
This script compare our Algorithm 2 with PDHG and SPDHG algorithms in the following strongly convex example:

    Soft margin SVM without bias
    min_{x, x0} sum_{j=1,...,m} f[j](b_j * (A[j,:] * x) ) + g(x),
    where f[j](r) = max(0 , 1 - r), g(x) = beta/2 norm(x)^2.

with the following data:

    "w8a", "rcv1", "real-sim", and "news20" from the LIBSVM datasets website.

If you want to test more SVM data:

    add your data under  the "./data/" folder.

By default the whole dimension is separated as 32 blocks.

If you want to quickly take a look at the performance of the algorithms:

    run the last part of this script: "Plot the result".

References:

    [1] A. Chambolle, M. J. Ehrhardt, P. Richtárik, and C.-B. Schönlieb. Stochastic primal-dual hybrid gradient
    algorithm with arbitrary sampling and imaging applications. SIAM J. Optim., 28(4):2783–2808, 2018.

    [2] T. Goldstein, E. Esser, and R. Baraniuk. Adaptive primal-dual hybrid gradient methods for saddle point
    problems. Tech. Report., pages 1–26, 2013. http://arxiv.org/pdf/1305.0546v1.pdf.
"""

from __future__ import division, print_function

import numpy as np
import math
from scipy import sparse as sp
import sys
from argParse import argParser_block
from solver_block.original_alg2 import original_alg2
from solver_block.spdhg import spdhg
from solver_block.pdhg import pdhg
from solver_block.spdc import spdc
from sklearn.datasets import load_svmlight_file

# Read Parameters
num_blk, data_name, num_epoch = argParser_block()
if data_name is None:
    data_name = "rcv1"

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

# fix the initial point and iterations
x0 = np.zeros(d)
niter = num_blk*num_epoch

# partition into blk_x blocks
blk_x = num_blk
bat_size_x = d // blk_x + 1
Ablk_x = []
index_x = []
for k in range(blk_x):
    mask = range((k % blk_x) * bat_size_x, min((k % blk_x + 1) * bat_size_x, d))
    index_x.append(mask)
    Ablk_x.append(A[:, mask])
Atblk_x = [Ablk_x[i].transpose() for i in range(blk_x)]

# partition into blk_x blocks
blk_r = num_blk
bat_size_r = m // blk_r + 1
Ablk_r = []
index_r = []
for k in range(blk_r):
    mask = range((k % blk_r) * bat_size_r, min((k % blk_r + 1) * bat_size_r, m))
    index_r.append(mask)
    Ablk_r.append(A[mask, :])
Atblk_r = [Ablk_r[i].transpose() for i in range(blk_r)]

# Define the solution set
Sol = dict()


# Define proximal operators
def proximal_x(x, sigma):
    return x / (beta * sigma + 1)


def proximal_r(r, sigma):
    return np.minimum(np.maximum(r, 1), r + sigma)


def proximal_r_star(r, sigma):
    return np.minimum(np.maximum(r - sigma, -1), 0)


"""
Run our Algorithm 2
"""
if_average = 1
tau0 = 1 / blk_x  # 1/(blk_x*blk_r)
rho0 = 5 / sp.linalg.norm(A)  # penalty parameter for r , 8 for w8a
sigma = 5 / sp.linalg.norm(A)  # penalty parameter for x, 8 for w8a
c_eta = 1  # 1 for w8a
c_tau = 2 / tau0  # 2 for w8a
if data_name == "w8a":
    rho0 = 8 / sp.linalg.norm(A)  # penalty parameter for r
    sigma = 8 / sp.linalg.norm(A)  # penalty parameter for x
    c_eta = 1
    c_tau = 2 / tau0

Sol["original_alg2"] = original_alg2(x0, proximal_x, proximal_r_star, A, Ablk_x, Atblk_x, index_x, if_average,
                                     niter,
                                     primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                     rho0=rho0, tau0=tau0, sigma=sigma, c_tau=c_tau, c_eta=c_eta, verbose=1,
                                     print_dist=int(max(niter / 20, 1)))
"""
Run PDHG algorithm
"""
gamma = 0.99
tau = gamma / sp.linalg.norm(A)
sigma = 0.01  # [gamma/sp.linalg.norm(A[i,:]) for i in range(0,m)]
if data_name == "w8a":
    sigma = 0.005
if_average = 1
Sol["pdhg"] = pdhg(x0, proximal_x, proximal_r_star, A, tau, sigma, if_average, num_epoch,
                   primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                   verbose=1, print_dist=int(max(num_epoch / 20, 1)))
"""
Run SPDHG algorithm
"""


def proximal_r_star(mask, r, sigma):
    return np.minimum(np.maximum(r - sigma, -1), 0)


gamma1 = 5
gamma2 = 5
tau = gamma1 / sp.linalg.norm(A)
sigma = gamma2 / sp.linalg.norm(A)
if_average = 1
theta = 1
Sol["spdhg"] = spdhg(x0, proximal_x, proximal_r_star, A, Ablk_r, Atblk_r, index_r, tau, sigma, if_average, niter,
                     primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                     theta=theta, verbose=1, print_dist=int(max(niter / 20, 1)))

"""
Save data
"""
import pickle

data = dict(dim_dual=m, dim_primal=d, blk_x=blk_x, blk_r=blk_r)

# calculate the initial primal value
x0 = np.zeros(d)
y0 = np.zeros(m)
primal_obj_x0 = primal_obj_func(x0)
dual_obj_y0 = dual_obj_func(y0)
gap0 = primal_obj_x0 - dual_obj_y0

Sol["spdhg"]["gap"] = [gap0] + Sol["spdhg"]["gap"]
Sol["pdhg"]["gap"] = [gap0] + Sol["pdhg"]["gap"]
Sol["original_alg2"]["gap"] = [gap0] + Sol["original_alg2"]["gap"]
data["sol"] = Sol
pickle.dump(data, open("result/svm/svm_" + data_name, "wb"))

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

# load the data
if 'data_name' not in locals():
    data_name = "rcv1"
data = pickle.load(open("result/svm/svm_" + data_name, "rb"))
m = data["dim_dual"]
d = data["dim_primal"]
blk_x = data["blk_x"]
blk_r = data["blk_r"]
total_iter = len(data["sol"]["original_alg2"]["gap"])
iters = np.array(range(0, total_iter))
num_epoch = int(total_iter / blk_r)

gap = np.array(data["sol"]["original_alg2"]["gap"]) / m
plt.semilogy(iters / blk_x, gap, 'C1-', marker='o', markevery=int(total_iter / mark_num),
             markersize=markersize, label=r'Original Algorithm 2')

gap = np.array(data["sol"]["spdhg"]["gap"]) / m
plt.semilogy(iters / blk_x, gap, 'C3-', marker='v', markevery=int(total_iter / mark_num),
             markersize=markersize, label=r'SPDHG')

gap = np.array(data["sol"]["pdhg"]["gap"]) / m
mask = np.array(range(num_epoch))
plt.semilogy(mask, gap[mask], 'C4-', marker='*', markevery=int(num_epoch / mark_num),
             markersize=markersize, label=r'PDHG')

plt.title("SVM - " + data_name + ": m = " + str(m) + ", n = " + str(d))
plt.xlabel("Epochs")
plt.ylabel("Duality Gap")
plt.legend()
plt.show()
