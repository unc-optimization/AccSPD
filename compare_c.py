"""
This script verifies that choosing different c for Algorithm 1 and Algorithm 2 in Non-strongly convex case
will cause different convergence rate. We illustrate this point by using the following example:

    Soft margin SVM without bias
    min_{x, x0} sum_{j=1,...,m} f[j](b_j * (A[j,:] * x) ) + g(x),
    where f[j](r) = c_j * max(0 , 1 - r), g(x) = beta/2 norm(x)^2

with the following data:

    "w8a" from the LIBSVM data set website.

If you want to test more SVM data:

    add your data under  the "./data/" folder.

Notice that this script could run a long time if number of epoch is big.
"""

from __future__ import division, print_function
import numpy as np
import math
from scipy import sparse as sp
import time
from argParse import argParser_block
from sklearn.datasets import load_svmlight_file
from solver_batch.original_alg1 import original_alg1
from solver_block.original_alg2 import original_alg2

# Read Parameters
num_blk, data_name, num_epoch = argParser_block()
if data_name is None:
    data_name = "w8a"

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
blk_x = num_blk
bat_size_x = d // blk_x + 1
Ablk_x = []
index_x = []
for k in range(blk_x):
    mask = range((k % blk_x) * bat_size_x, min((k % blk_x + 1) * bat_size_x, d))
    index_x.append(mask)
    Ablk_x.append(A[:, mask])
Atblk_x = [Ablk_x[i].transpose() for i in range(blk_x)]

"""
Run Algorithm 2 with different c_tau
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
rho0 = 5 / sp.linalg.norm(A)  # penalty parameter for r
sigma = 5 / sp.linalg.norm(A)  # penalty parameter for x
c_eta = 2

# case 1: c_tau*tau0 = 1
c_tau = 1 / tau0
Sol["original_alg2_case1"] = original_alg2(x0, proximal_x, proximal_r_star, A, Ablk_x, Atblk_x, index_x, if_average,
                                           niter,
                                           primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                           rho0=rho0, tau0=tau0, sigma=sigma, c_tau=c_tau, c_eta=c_eta, verbose=1,
                                           print_dist=500)
# case 2: c_tau*tau0 = 2
c_tau = 2 / tau0
Sol["original_alg2_case2"] = original_alg2(x0, proximal_x, proximal_r_star, A, Ablk_x, Atblk_x, index_x, if_average,
                                           niter,
                                           primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                           rho0=rho0, tau0=tau0, sigma=sigma, c_tau=c_tau, c_eta=c_eta, verbose=1,
                                           print_dist=500)
"""
Run Algorithm 1 with different c_tau
"""


# Define proximal operators
def proximal_x(j, x, sigma):
    return x / (beta * sigma + 1)


def proximal_r(r, sigma):
    return np.minimum(np.maximum(r, 1), r + sigma)


def proximal_r_star(i, r, sigma):
    return min(max(r - sigma, -1), 0)


# # Define batch size
# blk_x = 8
# def fun_select_x(k):
#     return np.random.choice(d, int(d/blk_x), replace = False)
# blk_r = 8
# def fun_select_r(k):
#     return np.random.choice(m, int(m/blk_r), replace = False)

# use cyclic
blk_x = num_blk
bat_size_x = d // blk_x + 1


def fun_select_x(k):
    return range((k % blk_x) * bat_size_x, min((k % blk_x + 1) * bat_size_x, d))


blk_r = num_blk
bat_size_r = m // blk_r + 1


def fun_select_r(k):
    return range((k % blk_r) * bat_size_r, min((k % blk_r + 1) * bat_size_r, m))


# set number of iteration
niter = int(blk_r * blk_x * num_epoch)

if_average = 1
tau0 = min(1 / blk_x, 1 / blk_r)  # 1/(blk_x*blk_r)
rho0 = 3 / sp.linalg.norm(A)  # penalty parameter for r
sigma = 3 / sp.linalg.norm(A)  # penalty parameter for x
c_eta = 2

# case1: c_tau*tau0 = 1
c_tau = 1 / tau0
Sol["original_alg1_case1"] = original_alg1(x0, proximal_x, proximal_r_star, A, if_average, niter,
                                           primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                           fun_select_x=fun_select_x, fun_select_r=fun_select_r,
                                           rho0=rho0, tau0=tau0, sigma=sigma, c_tau=c_tau, c_eta=c_eta, verbose=1,
                                           print_dist=500)

# case2: c_tau*tau0 = 2
c_tau = 2 / tau0
Sol["original_alg1_case2"] = original_alg1(x0, proximal_x, proximal_r_star, A, if_average, niter,
                                           primal_obj_func=primal_obj_func, dual_obj_func=dual_obj_func,
                                           fun_select_x=fun_select_x, fun_select_r=fun_select_r,
                                           rho0=rho0, tau0=tau0, sigma=sigma, c_tau=c_tau, c_eta=c_eta, verbose=1,
                                           print_dist=500)
"""
save data
"""
import pickle

data = dict(dim_dual=m, dim_primal=d, blk_x=blk_x, blk_r=blk_r)

# calculate the initial gap
x0 = np.zeros(d)
y0 = np.zeros(m)
primal_obj_x0 = primal_obj_func(x0)
dual_obj_y0 = dual_obj_func(y0)
gap0 = primal_obj_x0 - dual_obj_y0

Sol["original_alg1_case1"]["gap"] = [gap0] + Sol["original_alg1_case1"]["gap"]
Sol["original_alg1_case2"]["gap"] = [gap0] + Sol["original_alg1_case2"]["gap"]
Sol["original_alg2_case1"]["gap"] = [gap0] + Sol["original_alg2_case1"]["gap"]
Sol["original_alg2_case2"]["gap"] = [gap0] + Sol["original_alg2_case2"]["gap"]

data["sol"] = Sol
pickle.dump(data, open("result/compare_c/svm_" + data_name, "wb"))

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
    data_name = "w8a"
data = pickle.load(open("result/compare_c/svm_" + data_name, "rb"))
m = data["dim_dual"]
d = data["dim_primal"]
blk_x = data["blk_x"]
blk_r = data["blk_r"]

# create the figure
fig_compare_c = plt.figure(figsize=(20, 6))

# plot algorithm 1
inter_iter = blk_x * blk_r
total_iter = len(data["sol"]["original_alg1_case1"]["gap"])
xticks = range(inter_iter, total_iter, inter_iter)
mask = np.array(range(inter_iter, total_iter))
num_epoch = len(xticks)

fig_alg1 = plt.subplot(1, 2, 1)
gap = np.array(data["sol"]["original_alg1_case1"]["gap"]) / m
num_points = len(mask)
num_mark = 20
markers_on = [int(math.exp(i * math.log(num_points) / num_mark)) for i in range(2, num_mark)]
plt.loglog(mask / inter_iter, gap[mask], 'C1-', marker='o', markevery=markers_on,
           markersize=10, label=r'Algorithm 1 ($c\tau_0 = 1$)')

gap = np.array(data["sol"]["original_alg1_case2"]["gap"]) / m
plt.loglog(mask / inter_iter, gap[mask], 'b-', marker='s', markevery=markers_on,
           markersize=10, label=r'Algorithm 1 ($c\tau_0 = 2$)')

# # plot 1/k and 1/k^2
num_points = num_epoch
num_mark = 20
markers_on = [int(math.exp(i * math.log(num_points) / num_mark)) for i in range(2, num_mark)]
plt.loglog(range(1, num_epoch + 1), gap[inter_iter] / np.array(range(1, num_epoch + 1)),
           'C2-', marker='v', markevery=markers_on, markersize=10, label=r'$\mathcal{O}(1/K)$')
plt.loglog(range(1, num_epoch + 1), gap[inter_iter] / np.array(range(1, num_epoch + 1)) ** 2,
           'C3-', marker='*', markevery=markers_on, markersize=10, label=r'$\mathcal{O}(1/K^2)$')

fig_alg1.set_title("Compare with Different c in Algorithm 1 - " + data_name + \
                   ": m = " + str(m) + ", n = " + str(d))
plt.xlim(1, num_epoch + 1)
plt.xlabel("Epochs")
plt.ylabel("Duality Gap")
plt.legend(loc='lower left')

# plot algorithm 2
inter_iter = blk_x
total_iter = len(data["sol"]["original_alg2_case1"]["gap"])
xticks = range(inter_iter, total_iter, inter_iter)
mask = np.array(range(inter_iter, total_iter))
num_epoch = len(xticks)

fig_alg2 = plt.subplot(1, 2, 2)
gap = np.array(data["sol"]["original_alg2_case1"]["gap"]) / m
num_points = len(mask)
num_mark = 20
markers_on = [int(math.exp(i * math.log(num_points) / num_mark)) for i in range(2, num_mark)]
plt.loglog(mask / inter_iter, gap[mask], 'C1-', marker='o', markevery=markers_on,
           markersize=10, label=r'Algorithm 2 ($c\tau_0 = 1$)')

gap = np.array(data["sol"]["original_alg2_case2"]["gap"]) / m
plt.loglog(mask / inter_iter, gap[mask], 'b-', marker='s', markevery=markers_on,
           markersize=10, label=r'Algorithm 2 ($c\tau_0 = 2$)')

# # plot 1/k and 1/k^2
num_points = num_epoch
num_mark = 20
markers_on = [int(math.exp(i * math.log(num_points) / num_mark)) for i in range(2, num_mark)]
plt.loglog(range(1, num_epoch + 1), gap[inter_iter] / np.array(range(1, num_epoch + 1)),
           'C2-', marker='v', markevery=markers_on, markersize=10, label=r'$\mathcal{O}(1/K)$')
plt.loglog(range(1, num_epoch + 1), gap[inter_iter] / np.array(range(1, num_epoch + 1)) ** 2,
           'C3-', marker='*', markevery=markers_on, markersize=10, label=r'$\mathcal{O}(1/K^2)$')

fig_alg2.set_title("Compare with Different c in Algorithm 2 - " + data_name + \
                   ": m = " + str(m) + ", n = " + str(d))
plt.xlim(1, num_epoch + 1)
plt.xlabel("Epochs")
plt.ylabel("Duality Gap")
plt.legend(loc='lower left')
plt.show()
