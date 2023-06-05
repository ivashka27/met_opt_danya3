import numpy as np
from math import sqrt
from lab1.method import wolfe_gradient


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


def gauss_newton(f, jac, x, y, p0, eps=1e-4, max_iter=10000):
    jac_calc = 0
    func_calc = 0
    p = p0
    points = [np.asarray(p)]
    for itr in range(max_iter):
        J = jac(f(p), x)
        jac_calc += 1
        dy = y - f(p)(x)
        func_calc += 1
        new_p = p + np.linalg.inv(J.T @ J) @ J.T @ dy
        if np.linalg.norm(p - new_p) < eps:
            break
        p = new_p
        points.append(p)
    return np.asarray(points), func_calc, jac_calc


def dogleg_method(gk, Bk, trust_radius):
    pB = -np.dot(np.linalg.inv(Bk), gk)
    #print("pB:", pB)
    #print(np.dot(pB, pB))
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
    points = [xk]
    trust_radius = initial_trust_radius
    k = 0
    func_calc = 0
    jac_calc = 0
    while True:

        gk = jac(xk)
        jac_calc += 1
        Bk = hess(xk)

        pk = dogleg_method(gk, Bk, trust_radius)

        # Actual reduction.
        act_red = f(xk) - f(xk + pk)
        func_calc += 2

        # Predicted reduction.
        pred_red = -(np.dot(gk, pk) + 0.5 * np.dot(pk, np.dot(Bk, pk)))

        # Rho.

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
        points.append(xk)
        # Check if the gradient is small enough to stop
        if np.linalg.norm(gk) < eps:
            break

        # Check if we have looked at enough iterations
        if k >= max_iter:
            break
        k = k + 1
    return np.asarray(points), 0, 0


def bfgs(f, grad, start, eps=1e-4, max_iter=10000):
    x = np.array(start)
    points = [x]
    grad_calc = 1
    func_calc = 0
    gx = grad(x)

    E = np.eye(len(x))
    H = E

    for epoch in range(max_iter):
        pk = -np.dot(H, gx)

        lr = 1
        fx = f(x)

        func_calc += 2
        grad_calc += 1
        while not wolfe_gradient.wolfe_conditions(f, fx, grad, gx, x, pk, lr):
            func_calc += 1
            grad_calc += 1
            lr = lr / 2

        xn = x + lr * pk
        s = xn - x
        x = xn

        gradn = grad(xn)
        grad_calc += 1

        y = gradn - gx
        gx = gradn
        ro = 1.0 / (np.dot(y, s))
        С1 = E - ro * s[:, np.newaxis] * y[np.newaxis, :]
        С2 = E - ro * y[:, np.newaxis] * s[np.newaxis, :]
        H = np.dot(С1, np.dot(H, С2)) + (ro * s[:, np.newaxis] * s[np.newaxis, :])

        points.append(x)
        if np.linalg.norm(gx) < eps:
            break
    return np.asarray(points), grad_calc, func_calc


def l_bfgs(f, grad, start, eps=1e-4, max_iterations=10000, m=10):
    xk = np.array(start)
    I = np.identity(len(xk))
    Hk = I
    grad_calc = 0
    func_calc = 0
    funcs = []
    grads = []
    points = [xk]

    def calculate_pk(H0, p):
        m_t = len(funcs)
        q = p
        a = np.zeros(m_t)
        b = np.zeros(m_t)
        for i in reversed(range(m_t)):
            s = funcs[i]
            y = grads[i]
            rho_i = float(1.0 / y.T.dot(s))
            a[i] = rho_i * s.dot(q)
            q = q - a[i] * y

        r = H0.dot(q)

        for i in range(m_t):
            s = funcs[i]
            y = grads[i]
            rho_i = float(1.0 / y.T.dot(s))
            b[i] = rho_i * y.dot(r)
            r = r + s * (a[i] - b[i])

        return r

    for i in range(max_iterations):
        # compute search direction
        gk = grad(xk)
        grad_calc += 1
        pk = -calculate_pk(I, gk)

        fx = f(xk)
        func_calc += 2
        grad_calc += 1
        # obtain step length by line search
        lr = 1
        while not wolfe_gradient.wolfe_conditions(f, fx, grad, gk, xk, pk, lr):
            func_calc += 1
            grad_calc += 1
            lr = lr / 2

        # update x
        xk1 = xk + lr * pk
        gk1 = grad(xk1)
        grad_calc += 1

        # define sk and yk for convenience
        sk = xk1 - xk
        yk = gk1 - gk

        funcs.append(sk)
        grads.append(yk)
        if len(funcs) > m:
            funcs = funcs[1:]
            grads = grads[1:]

        # compute H_{k+1} by BFGS update
        # rho_k = float(1.0 / yk.dot(sk))

        #print(xk)
        points.append(xk)

        if np.linalg.norm(xk1 - xk) < eps:
            xk = xk1
            return np.asarray(points), grad_calc, func_calc

        xk = xk1
