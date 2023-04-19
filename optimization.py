import numpy as np
import copy

def Wolfe_step(struct_copy, original_X, alpha, pk):
    next_x = original_X+alpha*pk
    struct_copy.update_nodes(next_x)
    next_value = struct_copy.E()
    next_grad = struct_copy.gradient()
    return next_x, next_value, next_grad

def Wolfe(struct, pk, fk, grad_k, c1=1e-4, c2=0.9, alpha=1, maxiter=200):
    alpha_min = 0
    alpha_max = np.inf
    struct_copy = copy.deepcopy(struct)
    original_X = struct.X.copy() #Ensures that we get a deep copy

    next_x, next_f, next_grad = Wolfe_step(struct_copy, original_X, alpha, pk)
    initial_descent = np.dot(grad_k, pk)

    Armijo = next_f <=fk + c1*alpha*initial_descent
    curve_cond = np.dot(next_grad, pk) >= c2*initial_descent

    niter=0
    while niter<maxiter:
        if not Armijo:
            alpha_max = alpha
            alpha = (alpha_min+alpha_max)/2
        elif not curve_cond:
            alpha_min = alpha
            if alpha_max == np.inf:
                alpha *= 2
            else:
                alpha = (alpha_min+alpha_max)/2
        else:
            return next_x, next_f, next_grad

        next_x, next_f, next_grad = Wolfe_step(struct_copy, original_X, alpha, pk)

        Armijo = next_f <= fk + c1*alpha*initial_descent
        curve_cond = np.dot(next_grad, pk) >= c2*initial_descent

        niter += 1
    #print("Step length did not converge in Wolfe")
    return next_x, next_f, next_grad

def BFGS(struct,
         tol=1e-12,
         maxiter = 10000,
         alpha = 1.0,
         c1 = 1e-2,
         c2 = 0.9,
         max_extrapolation_iterations = 50,
         max_interpolation_iterations = 20,
         rho = 2.0,
         return_norms = False,
         return_images = False):
    """
    Calculates a stable configuration for the structure, using BFGS
    and Wolfe conditions
    """
    X = struct.X
    H = np.eye(X.size)
    iter = 0
    E_val = struct.E()
    grad = struct.gradient()
    norm = np.linalg.norm(grad)
    if return_norms:
        norms = [norm]
    if return_images:
        imgs = []
    while norm>tol and iter<maxiter:
        iter+=1
        if return_images:
            imgs.append(plot(struct)[0])
        p = -H @ grad #finds descent direction

        X_old = X.copy()  #saves previous X-value and gradient
        grad_old = grad.copy()


        X, E_val, grad = StrongWolfe(struct, p, E_val, grad, alpha,c1,c2, max_extrapolation_iterations, max_interpolation_iterations, rho)
        struct.update_nodes(X) #updates nodes of the structure

        if np.array_equal(X, X_old) or np.array_equal(grad, grad_old):
            print("LIKE")
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
    if return_images:
        if return_norms:
            return imgs, np.array(norms)
    if return_norms:
        return np.array(norms)

def NextStep(struct_copy, original_X, alpha, p):
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
    struct_copy = copy.deepcopy(struct)
    initial_descent = np.dot(grad, p)
    original_X = struct.X.copy()

    # initialise the bounds of the bracketing interval
    alphaR = alpha
    alphaL = 0.0

    next_x, next_E, next_grad = NextStep(struct_copy, original_X, alphaR, p)

    Armijo = (next_E <= E_val+c1*alphaR*initial_descent)
    descentR = np.inner(p, next_grad)
    curvatureLow = (descentR >= c2*initial_descent)
    curvatureHigh = (descentR <= -c2*initial_descent)

    iter = 0
    # We start by increasing alphaR as long as Armijo and curvatureHigh hold,
    # but curvatureLow fails (that is, alphaR is definitely too small).
    # Note that curvatureHigh is automatically satisfied if curvatureLow fails.
    # Thus we only need to check whether Armijo holds and curvatureLow fails.
    while (iter < max_extrapolation_iterations and (Armijo and (not curvatureLow))):
        alphaL = alphaR
        alphaR *= rho

        # update function value and gradient
        next_x, next_E, next_grad = NextStep(struct_copy, original_X, alphaR, p)

        # update the Armijo and Wolfe conditions

        Armijo = (next_E <= E_val+c1*alphaR*initial_descent)
        descentR = np.inner(p, next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)

        iter += 1
    # at that point we should have a situation where alphaL is too small
    # and alphaR is either satisfactory or too large
    # (Unless we have stopped because we used too many iterations. There
    # are at the moment no exceptions raised if this is the case.)
    alpha = alphaR
    iter = 0
    # Use bisection in order to find a step length alpha that satisfies
    # all conditions.
    while (iter < max_interpolation_iterations and (not (Armijo and curvatureLow and curvatureHigh))):
        if (Armijo and (not curvatureLow)):
            # the step length alpha was still too small
            # replace the former lower bound with alpha
            alphaL = alpha
        else:
            # the step length alpha was too large
            # replace the upper bound with alpha
            alphaR = alpha
        iter += 1

        # choose a new step length as the mean of the new bounds
        alpha = (alphaL+alphaR)/2
        # update function value and gradient
        next_x, next_E, next_grad = NextStep(struct_copy, original_X, alphaR, p)
        # update the Armijo and Wolfe conditions
        Armijo = (next_E <= E_val+c1*alphaR*initial_descent)
        descentR = np.inner(p, next_grad)
        curvatureLow = (descentR >= c2*initial_descent)
        curvatureHigh = (descentR <= -c2*initial_descent)
    # return the next iterate as well as the function value and gradient there
    # (in order to save time in the outer iteration; we have had to do these
    # computations anyway)
    #if np.array_equal(next_x, original_X):
     #   print("x-ene er like!!!!")
      #  print(next_x.dtype)
       # print("alpha*p", alphaR*p)
        #print("next_x", next_x)
        #print("original_x", original_X)
        #print("next_x??", original_X + alphaR*p)
        #print("p:", p)
        #print("alpha:", alphaR)
    return next_x, next_E, next_grad


def quadratic_penalty_method(struct, penalty0, tolerances, maxiter_BFGS = 50, TOL=1e-12, max_penalty = 1e6):
    #struct_copy = copy.deepcopy(struct)
    #struct_copy.penalty = penalty0
    struct.penalty = penalty0
    K = tolerances.size
    for k in range(K):
        BFGS(struct, tol=tolerances[k], maxiter=maxiter_BFGS)
        norm_grad = np.linalg.norm(struct.gradient())

        if norm_grad <= TOL:
            break

        if struct.penalty < max_penalty:
            struct.penalty *= 10


"""
def barrier_method(struct, penalty0, tolerances, maxiter_BFGS = 50, TOL=1e-12, max_penalty = 1e6):
    struct.penalty = penalty0
    K = tolerances.size
    for k in range(K):
        BFGS(struct, tol=tolerances[k], maxiter=maxiter_BFGS)
        norm_grad = np.linalg.norm(struct.gradient())

        if norm_grad <= TOL:
            break

        if struct.penalty < max_penalty:
            struct.penalty /= 10
"""