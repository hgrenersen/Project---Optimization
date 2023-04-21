import numpy as np
import copy

def BFGS(struct,
         tol=1e-12,
         maxiter = 10000,
         alpha = 1.0,
         c1 = 1e-2,
         c2 = 0.9,
         max_extrapolation_iterations = 50,
         max_interpolation_iterations = 20,
         rho = 2.0,
         return_norms = False):
    """
    Calculates a stable configuration for the structure, using BFGS
    and Wolfe conditions.
    """
    X = struct.X
    H = np.eye(X.size)
    iter = 0
    E_val = struct.E()
    grad = struct.gradient()
    norm = np.linalg.norm(grad)
    if return_norms:
        norms = [norm]
    while norm>tol and iter<maxiter:
        iter+=1
    
        p = -H @ grad #finds descent direction

        X_old = X.copy()  #saves previous X-value 
        grad_old = grad.copy() #saves previous gradient


        X, E_val, grad = StrongWolfe(struct, p, E_val, grad, alpha,c1,c2, max_extrapolation_iterations, max_interpolation_iterations, rho)
        struct.update_nodes(X) #updates nodes of the structure

        if np.array_equal(X, X_old) or np.array_equal(grad, grad_old):
            break

        s = X-X_old
        y = grad-grad_old

        r = 1/np.inner(y, s)

        if iter == 0:
            H *= 1/(r*np.inner(y, y))
        z = H @ y
        H += -r*(np.outer(s, z) + np.outer(z, s)) + r*(r*np.inner(y, z)+1)*np.outer(s, s)

        norm = np.linalg.norm(grad)
        if return_norms:
            norms.append(norm)

    print(f"BFGS used {iter} iterations")
    if return_norms:
        return np.array(norms)

def NextStep(struct_copy, original_X, alpha, p):
    """
    Helping function used in StrongWolfe() in order to
    make a step with a given step size. 
    """
    next_x = original_X+alpha*p
    struct_copy.update_nodes(next_x)
    next_E = struct_copy.E()
    next_grad = struct_copy.gradient().copy()
    return next_x, next_E, next_grad

def StrongWolfe(struct, p,
                E_val,
                grad,
                alpha = 1.0,
                c1 = 1e-2,
                c2 = 0.9,
                max_extrapolation_iterations = 50,
                max_interpolation_iterations = 20,
                rho = 2.0):
    """
    Implementation of Strong Wolfe conditions for step size selection. When 
    the step size is chosen, we return the next step, defined by p and this
    chosen step, as we had to calculate this next step in order to choose the
    step size. 
    This approach is inspired by the code provided in the exercies
    in the course.
    """
    struct_copy = copy.deepcopy(struct)
    initial_descent = np.dot(grad, p)
    original_X = struct.X.copy()

    # Making an interval for which we consider possible step sizes to lie in.
    # This will be updated throughout the code
    alphaR = alpha
    alphaL = 0.0

    # The next five following lines of code are repeated any time a step 
    # size is evaluated.
    # Making a step with our starting stepsize, alpha.
    next_x, next_E, next_grad = NextStep(struct_copy, original_X, alphaR, p)

    # Implementation of conditions we want to consider in order to 
    # determine our stepsize.
    Armijo = (next_E <= E_val+c1*alphaR*initial_descent)
    descentR = np.inner(p, next_grad)
    curvatureLow = (descentR >= c2*initial_descent)
    curvatureHigh = (descentR <= -c2*initial_descent)

    iter = 0

    # As long as alphaR is too small, we know that Armijo and curvatureLow fails
    #  (therefore also curvatureHigh fails), and we therefore increase our upper 
    # bound, while updating the lower bound to the current stepsize considered
    while (iter < max_extrapolation_iterations and (Armijo and (not curvatureLow))):
        alphaL = alphaR
        alphaR *= rho

        next_x, next_E, next_grad = NextStep(struct_copy, original_X, alphaR, p)

        Armijo = (next_E <= E_val+c1*alphaR*initial_descent)
        descentR = np.inner(p, next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)

        iter += 1
    # at that point we should have a situation where alphaL is too small
    # and alphaR is either satisfactory or too large
    # (Unless we have stopped because we used too many iterations. There
    # are at the moment no exceptions raised if this is the case.)

    # Now, hopefully, the only problem should be that alphaL is too small 
    # and potentially that alphaR is too high. Unless iter > max_extrapolation_
    # iterations

    alpha = alphaR
    iter = 0
    # Below we use bisection in order to choose our alpha
    while (iter < max_interpolation_iterations and (not (Armijo and curvatureLow and curvatureHigh))):
        if (Armijo and (not curvatureLow)):
            # Similarly to the previous situation, the stepsizes are 
            # too small and we rise our lower bound to our current stepsize
            alphaL = alpha
        else:
            # The other alternative, alpha is too large so we decrease
            # the upper bound
            alphaR = alpha
        iter += 1

        # The mean of our interval is then chosen as the current step size
        alpha = (alphaL+alphaR)/2
        
        next_x, next_E, next_grad = NextStep(struct_copy, original_X, alphaR, p)
        
        Armijo = (next_E <= E_val+c1*alphaR*initial_descent)
        descentR = np.inner(p, next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
  
    return next_x, next_E, next_grad


def quadratic_penalty_method(struct, penalty0, tolerances, maxiter_BFGS = 50, TOL=1e-12, max_penalty = 1e6, return_norms = False):
    """
    Uses BFGS to calculate more stable configurations for different penalties.
    It is important that the initial penalty is high and that BFGS is allowed
    sufficient iterations in order to get a good starting point for later BFGS
    runs with higher penalties. This is of course trivial, and some tuning of
    parameters will likely be needed.
    """
    struct.penalty = penalty0
    K = tolerances.size
    if return_norms:
        norms_tot = np.array([])
    for k in range(K):
        if return_norms:
            norms = BFGS(struct, tol=tolerances[k], maxiter=maxiter_BFGS, return_norms=True)
            norms_tot = np.concatenate((norms_tot, norms))
        else:
            BFGS(struct, tol=tolerances[k], maxiter=maxiter_BFGS)
        norm_grad = np.linalg.norm(struct.gradient())

        if norm_grad <= TOL:
            break

        if struct.penalty < max_penalty:
            struct.penalty *= 10
    if return_norms:
        return norms_tot
