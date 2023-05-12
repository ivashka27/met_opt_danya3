import numpy as np
from math import sqrt


def jacobian(function, x):
    n = len(x)
    eps = 1e-6
    jacobian_matrix = np.zeros((n, n))
    for i in range(n):
        x_plus = np.copy(x)
        x_plus[i] += eps
        y_plus = function(x_plus)
        x_minus = np.copy(x)
        x_minus[i] -= eps
        y_minus = function(x_minus)
        jacobian_matrix[:, i] = (y_plus - y_minus) / (2 * eps)
    return jacobian_matrix


def gauss_newton(f, x, y, p0, eps, max_iter):
    p = p0
    for itr in range(max_iter):
        J = jacobian(f(p), x)
        dy = y - f(p)(x)
        new_p = p + np.linalg.inv(J.T @ J) @ J.T @ dy
        if np.linalg.norm(p - new_p) < eps:
            break
        p = new_p
    return p

def dogleg_method(gk, Bk, trust_radius):
    pB = -np.dot(np.linalg.inv(Bk), gk)
    norm_pB = sqrt(np.dot(pB, pB))

    if norm_pB <= trust_radius:
        return pB

    pU = - (np.dot(gk, gk) / np.dot(gk, np.dot(Bk, gk))) * gk
    dot_pU = np.dot(pU, pU)
    norm_pU = sqrt(dot_pU)

    if norm_pU >= trust_radius:
        return trust_radius * pU / norm_pU

    pB_pU = pB - pU
    dot_pB_pU = np.dot(pB_pU, pB_pU)
    dot_pU_pB_pU = np.dot(pU, pB_pU)
    fact = dot_pU_pB_pU ** 2 - dot_pB_pU * (dot_pU - trust_radius ** 2)
    tau = (-dot_pU_pB_pU + sqrt(fact)) / dot_pB_pU

    return pU + tau * pB_pU


def trust_region_dogleg(f, jac, hess, start, initial_trust_radius=1.0, max_trust_radius=100.0, eta=0.15, eps=1e-4,
                        max_iter=1000):
    xk = start
    trust_radius = initial_trust_radius
    k = 0
    while True:

        gk = jac(xk)
        Bk = hess(xk)

        pk = dogleg_method(gk, Bk, trust_radius)

        # Actual reduction.
        act_red = f(xk) - f(xk + pk)

        # Predicted reduction.
        pred_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))

        # Rho.
        rhok = act_red / pred_red
        if pred_red == 0.0:
            rhok = 1e99
        else:
            rhok = act_red / pred_red

        # Calculate the Euclidean norm of pk.
        norm_pk = sqrt(np.dot(pk, pk))

        # Rho is close to zero or negative, therefore the trust region is shrunk.
        if rhok < 0.25:
            trust_radius = 0.25 * norm_pk
        else:
            # Rho is close to one and pk has reached the boundary of the trust region, therefore the trust region is expanded.
            if rhok > 0.75 and norm_pk == trust_radius:
                trust_radius = min(2.0 * trust_radius, max_trust_radius)
            else:
                trust_radius = trust_radius

        # Choose the position for the next iteration.
        if rhok > eta:
            xk = xk + pk
        else:
            xk = xk

        # Check if the gradient is small enough to stop
        if np.linalg.norm(gk) < eps:
            break

        # Check if we have looked at enough iterations
        if k >= max_iter:
            break
        k = k + 1
    return xk