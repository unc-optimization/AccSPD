from __future__ import print_function, division
import numpy as np
import time

"""
Stochastic Primal-Dual Coordinate (SPDC) algorithms
"""


def spdc(x, proximal_x, proximal_y, A, Ablk, Atblk, index, tau, sigma, theta, if_average, niter, **kwargs):
    """Computes a saddle point with a SPDC.
    This means, a solution (x*, y*), y* = (y*_1, ..., y*_n) such that
    (x*, y*) in arg min_x max_y sum_i=1^m <y_i, A_i x> - f*_i(y_i) + g(x),
    where A in R^m*n, A_i is the ith row of A.
    g and f*_i are convex functions.
    For this algorithm, they all may be non-smooth and no strong convexity is assumed.
    Parameters
    ----------
    x : primal variable.
    proximal_x : proximal operator of objective function g of x
    proximal_y : proximal operator of objective function f* of y
    A : matrix in R^m*n
    Ablk: a list whose element is matrix. It define the partition of A
    Atblk: a list whose element is the transpose of the element of Ablk
    index: a list whose element is index. It define the partition of x
    tau : scalar, step size for primal variable.
    sigma : vector, step size for dual variable.
    theta : scalar. extrapolation factor for x.
    if_average: if use the average of the output.
    niter : int, number of iterations
    Other Parameters
    ----------------
    y : dual variable. By default equals 0.
    z : intermediate variable, represents At*y.
    primal_obj_func : primal objective function.
    dual_obj_func : dual objective function.
    verbose: True or False. Print the information of each step if true, don't print otherwise.
    print_dist : int. Print information after every print_dist iterations
    References
    ----------
    Zhang, Yuchen and Xiao, Lin. Stochastic Primal-Dual Coordinate Method for
    Regularized Empirical Risk Mini-mization. InICML, 2015
    """

    # get the dimension
    m, n = A.shape

    # Dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = np.zeros(m)

    # Adjoint of dual variable
    z = kwargs.pop('z', None)
    if z is None:
        z = A.transpose().dot(y)

    # primal objective value of x
    # dual objective value of y
    primal_obj_func = kwargs.pop('primal_obj_func', None)
    if primal_obj_func is None:
        def primal_obj_func(x):
            return 0
    dual_obj_func = kwargs.pop('dual_obj_func', None)
    if dual_obj_func is None:
        def dual_obj_func(x):
            return 0

    # print setup
    verbose = kwargs.pop('verbose', None)
    print_dist = kwargs.pop('print_dist', 1)

    # Initialize variables
    x_bar = xx = x = x.copy()
    blk_r = len(Ablk)
    At = A.transpose()
    primal_obj_val = []
    dual_obj_val = []
    gap = []
    each_time = []
    total_time = 0

    # Print initial information
    if verbose:
        print('Start SPDC...')
        print(
            '{message:{fill}{align}{width}}'.format(message='', fill='=', align='^', width=100, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='Epoch', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Primal Obj', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Dual Obj', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Gap', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Each Time', fill=' ', align='^', width=13, ), '|',
            '{message:{fill}{align}{width}}'.format(message='Total Time', fill=' ', align='^', width=13, ), '\n',
            '{message:{fill}{align}{width}}'.format(message='', fill='-', align='^', width=100, )
        )

    # run the iterations
    for k in range(niter):

        # record the start time of each iteration
        start_time = time.time()

        # update dual variable
        mask = index[k % blk_r]
        y_old = y[mask]

        # temp = y_old[i] + sigma_i * (Ai*x)
        temp = y_old + sigma * (Ablk[k % blk_r].dot(x_bar))

        # update y
        y[mask] = y_new = proximal_y(temp, sigma)

        # update dz
        dz = Atblk[k % blk_r].dot(y_new - y_old)

        # update x, z
        x_old = x
        temp = x - tau * (z + dz)  # x - tau*(z + (m/batch_size)*dz)
        x = proximal_x(temp, tau)
        x_bar = x + theta * (x - x_old)
        z += dz

        if if_average:
            xx = (k / (k + 1)) * xx + (1 / (k + 1)) * x
        else:
            xx = x

        # store information
        primal_obj_val.append(primal_obj_func(xx))
        dual_obj_val.append(dual_obj_func(y))
        gap.append(primal_obj_val[-1] - dual_obj_val[-1])
        each_time.append(time.time() - start_time)
        total_time = total_time + each_time[-1]

        # print the information
        if verbose:
            if k % print_dist == 0:
                print(
                    '{:^14.4f}'.format(int(k/blk_r)), '|',
                    '{:^13.3e}'.format(primal_obj_val[-1]), '|',
                    '{:^13.3e}'.format(dual_obj_val[-1]), '|',
                    '{:^14.4e}'.format(gap[-1]), '|',
                    '{:^13.3f}'.format(each_time[-1]), '|',
                    '{:^13.3e}'.format(total_time), '|',
                )

    return dict(primal_opt=xx, dual_opt=y, primal_obj_val=primal_obj_val, dual_obj_val=dual_obj_val, gap=gap,
                each_time=each_time, total_time=total_time)
