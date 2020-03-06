import numpy as np
import cvxpy as cp
import scipy as scipy
import cvxopt as cvxopt

def trend_filtering(y, vlambda, order=1):
    
    n = len(y)
    e = np.ones((1, n))
    if order == 1:
        D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)
    elif order == 2:
        D = scipy.sparse.spdiags(np.vstack((e, -3*e, 3*e, -e)), range(4), n-3, n)
    else:
        raise Exception("order can only be 1 or 2")
    
    x = cp.Variable(shape=n)
    obj = cp.Minimize(0.5 * cp.sum_squares(y - x)
                      + vlambda * cp.norm(D*x, 1) )
    prob = cp.Problem(obj)

    prob.solve(solver=cp.OSQP)
    
    return x.value
