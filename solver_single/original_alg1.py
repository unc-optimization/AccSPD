""" Our Algorithm"""
from __future__ import print_function, division
import numpy as np
from scipy import sparse as sp
import time

def original_alg1(x, proximal_x, proximal_r, A, if_average, niter, **kwargs):
    """Computes x* = arg min_x { sum_j=1^n f_j(x_j) + sum_i=1^m g_i( A_i x)},
    where A in R^m*n, A_i is the i th row of A.
    g_i and f_j are convex functions.
    we can define the following augmented Lagrangian function:
    L(x,r,y) = f(x) + g(r) + <y, A x - r> + rho/2 ||A x - r||^2
    Parameters
    ----------
    x : primal variable
    proximal_x : proximal operator of objective function f of x.
        proximal_x(j, x, sigma) = arg min_u{sigma*f_j(u) + 1/2||u - x||^2}
    proximal_r : proximal operator of objective function g of r, r is the intermediate variable
        which represents A x. proximal_r(i, r, sigma) = arg min_u{sigma* g*_i(u) + 1/2||u - r||^2}
    A : matrix in R^m*n
    if_average : if averge the dual variable y.
    niter : int, Number of iterations
    Other Parameters
    ----------------
    y : dual variable, optional. By default equals 0.
    r : intermediate variable, optional. By default equals A x.
    primal_obj_func : primal objective function, return the primal objective value.
    dual_obj_func : dual objective function, return the dual objective value.
    rho0, tau0, sigma, c_tau, c_eta: parameters that will influence the spped of our algorithms.
    verbose: True or False, print the information of each step if true, don't print otherwise.
    print_dist : int, print information after every print_dist iterations.
    fun_select_x : Function that selects blocks of x at every iteration. By
        default this is serial uniform sampling, fun_select_x(k) selects an index
        i \in {1,...,n} with probability 1/n.
    fun_select_r : Function that selects blocks of r at every iteration. By
    default this is serial uniform sampling, fun_select_r(k) selects an index
    i \in {1,...,m} with probability 1/m.
    References
    ----------
    """
    
    
    # get the dimension
    m, n = A.shape
    
    # dual variable
    y = kwargs.pop('y', None)
    if y is None:
        y = np.zeros(m)

    # intermediate variable
    r = kwargs.pop('r', None)
    if r is None:
        r = A.dot(x)
    
    # parameters
    rho0 = kwargs.pop('rho0', 1e-2)
    tau0 = kwargs.pop('tau0', min(1/m,1/n))
    sigma = kwargs.pop('sigma', 1e-2)
    c_tau = kwargs.pop('c_tau', 1)
    c_eta = kwargs.pop('c_eta', 1)
        
    # primal objective function of x
    # dual objective function of y
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

    # Selection function
    fun_select_x = kwargs.pop('fun_select_x', None)
    if fun_select_x is None:
        def fun_select_x(k):
            return [int(np.random.choice(n, 1))]
    
    fun_select_r = kwargs.pop('fun_select_r', None)
    if fun_select_r is None:
        def fun_select_r(x):
            return [int(np.random.choice(m, 1))]

    # Initialize variables
    # Initialize variables
    tau = tau0
    rho = rho0
    eta = rho0/c_eta
    x = x.copy()
    x_hat = x_tilde = x
    Ax = Ax_hat = Ax_tilde = A.dot(x)
    diff = Ax - r
    yy_bar = yy = y
    At = A.transpose()
    primal_obj_val = []
    dual_obj_val = []
    gap = []
    each_time = []
    total_time = 0
 
    # Print initial information
    if verbose:
        print('Start Our Method Algorithm 1...')
        print(
        '{message:{fill}{align}{width}}'.format(message='',fill='=',align='^',width=100,),'\n',
        '{message:{fill}{align}{width}}'.format(message='Epoch',fill=' ',align='^',width=13,),'|',
        '{message:{fill}{align}{width}}'.format(message='Primal Obj',fill=' ',align='^',width=13,),'|',
        '{message:{fill}{align}{width}}'.format(message='Dual Obj',fill=' ',align='^',width=13,),'|',
        '{message:{fill}{align}{width}}'.format(message='Gap',fill=' ',align='^',width=13,),'|',
        '{message:{fill}{align}{width}}'.format(message='Each Time',fill=' ',align='^',width=13,), '|',
        '{message:{fill}{align}{width}}'.format(message='Total Time',fill=' ',align='^',width=13,), '\n',
        '{message:{fill}{align}{width}}'.format(message='',fill='-',align='^',width=100,)
        )
        
    # run the iterations
    for k in range(niter):
        
        # record the start time of each iteration
        start_time = time.time()

        # select block
        selected_x = fun_select_x(k)
        selected_r = fun_select_r(k)
                
        # get x_hat Ax_hat
        x = (1 - tau)*x + tau*x_tilde
        Ax_hat = (1 -tau)*Ax + tau*Ax_tilde
        
        for i in selected_r:
            
            # first update y and then update r
            temp = y[i] + rho*Ax_hat[i]
            yy[i] = proximal_r(i, temp, rho)
            r[i] = (temp - yy[i])/rho
        
        # initialize the difference of Ax_tilde
        diff_Ax_tilde = np.zeros_like(Ax_tilde)
            
        for j in selected_x:
            
            # update x_tilde
            partial_x = At[j,:].dot(y + rho*(Ax_hat - r))
            x_old = x_tilde[j]
            x_tilde[j] = proximal_x(j, x_old - sigma*partial_x, sigma)
                        
            # update x
            diff_Ax_tilde += A[:,j].dot([x_tilde[j] - x_old])
            x[j] = x[j] + (tau/tau0)*(x_tilde[j] - x_old)                        
            
        # update Ax_tilde Ax
        Ax_tilde = Ax_tilde + diff_Ax_tilde
        Ax = Ax_hat + (tau/tau0)*diff_Ax_tilde    
    
        # calculate the new difference, update the dual variable
        new_diff = Ax - r
        y = y + eta*(new_diff - (1-tau)*diff)

        # decide if use average
        if if_average:
            yy_bar = (1 - tau)*yy_bar + tau*yy
        else:
            yy_bar = yy

        # update tau rho eta
        tau = c_tau*tau0/(k + c_tau)#c_tau*tau0/(tau0*k + c_tau)
        rho = rho0*tau0/tau
        eta = rho/c_eta
        
        # update diff
        diff = new_diff
                
        # store information
        primal_obj_val.append(primal_obj_func(x))
        dual_obj_val.append(dual_obj_func(yy_bar))
        gap.append(primal_obj_val[-1] - dual_obj_val[-1])
        each_time.append(time.time() - start_time)
        total_time = total_time + each_time[-1]
    
        # print the information
        if verbose:
            if k % print_dist == 0:
                print(
                '{:^14.4f}'.format(k),'|',
                '{:^13.3e}'.format(primal_obj_val[-1]),'|',
                '{:^13.3e}'.format(dual_obj_val[-1]),'|',
                '{:^14.4e}'.format(gap[-1]),'|',
                '{:^13.3f}'.format(each_time[-1]),'|',
                '{:^13.3e}'.format(total_time),'|',
                )
        
    return {"primal_opt": x, "dual_opt": yy_bar, \
            "primal_obj_val": primal_obj_val, "dual_obj_val": dual_obj_val, "gap": gap, \
            "each_time": each_time, "total_time": total_time}
