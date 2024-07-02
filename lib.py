import numpy as np
import time

def hard_thresholding(x: np.ndarray, lmbda: float) -> np.ndarray:
    """ Hard thresholding function
    Returns 0 if |x| < lmbda, otherwise x

    Args:
        x: input array
        lmbda: threshold value

    Returns:
        thresholded array
    """
    return np.where(np.abs(x) < lmbda, 0, x)


def soft_thresholding(x: np.ndarray, lmbda: float) -> np.ndarray:
    """ Soft thresholding function
    Returns sign(x) * max(|x| - lmbda, 0)

    Args:
        x: input array
        lmbda: threshold value

    Returns:
        thresholded array
    """
    return np.sign(x) * np.maximum(np.abs(x) - lmbda, 0)

def get_psnr(src: np.ndarray, pred: np.ndarray) -> float:
    """ Compute PSNR between two images
    Input images should be in [0, 1] range
    Args:
        src: source image
        pred: predicted image

    Returns:
        PSNR value
    """
    mse = np.mean((src - pred)**2)
    return 10 * np.log10(1 / mse)


def OMP(s: np.ndarray, D: np.ndarray, L: int, min_residual_norm: float = 1e-3) -> np.ndarray:
    """Orthogonal Matching Pursuit algorithm
    Args:
        s: input signal
        D: dictionary
        L: sparsity level
        min_residual_norm: minimum residual norm to stop the iteration

    Returns:
        sparse representation of the input signal (l0 norm <= L)
    """
    N = D.shape[1]
    x = np.zeros(N)
    r = s.copy()
    omega = []
    r_norm = np.linalg.norm(r)


    while np.count_nonzero(x) <= L and r_norm > min_residual_norm:
        # SWEEP STEP: look for the column of D that matches at best noisySignal
        # compute the residual w.r.t. each column of D
        e = np.zeros(N)
        for j in range(N):
            e[j] = np.linalg.norm(D[:, j] @ r)

        # find the column of D that matches at best r
        jStar = np.argmax(e)

        # UPDATE the support set with the jStar coefficient
        omega.append(jStar)

        _omega = list(omega)

        # update the coefficients by solving the least square problem min ||D_omega x - s ||
        x_OMP = np.linalg.lstsq(D[:, _omega], s)[0]
        # update the residual
        r = s - D[:, _omega] @ x_OMP

        r_norm = np.linalg.norm(r)

        x = np.zeros(N)
        x[list(omega)] = x_OMP.reshape(-1)
    
    return x

def LSOMP(s: np.ndarray, D: np.ndarray, L: int, min_residual_norm: float = 1e-3) -> np.ndarray:
    """Least Squares Orthogonal Matching Pursuit algorithm
    Args:
        s: input signal
        D: dictionary
        L: sparsity level
        min_residual_norm: minimum residual norm to stop the iteration

    Returns:
        sparse representation of the input signal (l0 norm <= L)
    """
    N = D.shape[1]
    x_LSOMP = np.zeros(N)
    r = s.copy()
    omega = []
    r_norm = np.linalg.norm(r)


    while np.count_nonzero(x_LSOMP) <= L and r_norm > min_residual_norm:
        # SWEEP STEP: find the best column by solving the LS problem
        solutions = {}
        for j in np.setdiff1d(range(N), omega):
            omega_temp = np.array([*omega, j])
            z = np.linalg.lstsq(D[:, omega_temp], s, rcond=None)[0]
            e = np.linalg.norm(s - D[:, omega_temp] @ z)
            solutions[j] = {
                'sol': z,
                'error': e,
            }
        
        j_star = min(solutions, key=lambda j: solutions[j]['error'])

        omega.append(j_star)
        x_LSOMP = np.zeros(N)
        x_LSOMP[omega] = solutions[j_star]['sol'].reshape(-1)

        print(f'Chosen {j_star}\t with error: {solutions[j_star]["error"]}')

        # update the residual
        r = s - D[:, omega] @ x_LSOMP[omega]
        r_norm = np.linalg.norm(r)

    return x_LSOMP


def KSVD(S: np.ndarray, N: int, tau: float, sparsity: int, max_iter: int = 10) -> tuple[np.ndarray, np.ndarray]: 
    """KSVD algorithm
    Args:
        S: input signals
        N: number of atoms
        tau: error threshold for the OMP algorithm
        max_iter: maximum number of iterations
        sparsity: sparsity level

    Returns:
        dictionary and sparse representations
    """

    # intialize the dictionary
    D = np.random.randn(S.shape[0], N)
    n_patch = S.shape[1]

    # normalize each column of D (zero mean and unit norm)
    # UPDATE D
    for j in range(N):
        D[:, j] = D[:, j] - D[:, j].mean()
        D[:, j] = D[:, j] / np.linalg.norm(D[:, j])

    # initialize the coefficient matrix
    X = np.zeros((N, n_patch))

    for iter in range(max_iter):
        time_start = time.time()
        print(f'Iteration {iter+1}/{max_iter}')

        # perform the sparse coding via OMP of all the columns of S
        for n in range(n_patch):
            X[:, n] = OMP(S[:, n], D, sparsity, tau)

        # iterate over the columns of D
        for j in range(N):
            # find which signals uses the j-th atom in the sparse coding
            omega = np.where(X[j, :] != 0)[0]

            if len(omega) == 0:
                continue
                # if the atom is never used then ignore or substitute it with a random vector
            else:
                # compute the residual matrix E, ignoring the j-th atom
                E = S
                for j0 in range(N):
                    if(j0 != j):
                        E = E - D[:, j0].reshape(-1 ,1) @ X[j0, :].reshape(1, -1)

                # restrict E to the columns indicated by omega
                Eomega = E[:, omega]

                # compute the SVD of Eomega
                U, Sigma, V = np.linalg.svd(Eomega)

                # update the dictionary
                D[:, j] = U[:, 0]

                # update the coefficient matrix
                X[j, omega] = Sigma[0] * V[0, :]

        time_end = time.time()
        print(f'This iteration took {time_end - time_start:.0f}s')

    return D, X

def ISTA(f: callable, df: callable, x0: np.ndarray, lmbda: float, max_iter: int = 1000, tol: float = 1e-3, tol_grad_norm: float = 1e-4, verbose = False) -> np.ndarray:
    """Iterative Soft Thresholding Algorithm
    Args:
        f: function to minimize
        df: gradient of the function
        x0: initial guess
        lmbda: regularization parameter
        max_iter: maximum number of iterations
        tol: tolerance for the stopping criterion
        tol_grad_norm: tolerance for the gradient norm

    Returns:
        solution of the optimization problem
    """

    # optimal value for alpha
    gamma = 1
    grad_norm = 1e10
    distanceX = 1e10

    # initialize the list with all the estimates
    all_x = [x0]

    cnt = 0

    x = x0.copy()

    while cnt < max_iter and distanceX > tol and grad_norm > tol_grad_norm:
        # compute the argument of the proximal operator
        x = x - gamma * df(x)

        # perform soft thresholding of x
        x = soft_thresholding(x, gamma * lmbda)

        # compute the norm of the gradient for the stopping criteria
        grad_norm = np.linalg.norm(df(x))

        # compute the distance between two consecutive iterates for the stopping criteria
        distanceX = np.linalg.norm(all_x[-1] - x)

        # store the estimate
        all_x += [x]

        cnt += 1

        if verbose:
            print(f'Iteration {cnt}/{max_iter}\t Gradient norm: {grad_norm:.2f}\t Distance: {distanceX:.2f}')

    if verbose:
        print(f'Converged after {cnt} iterations')
        print(f'The minimum is {f(x):.2f}\t at x = {x}')
    return x


def FISTA(f: callable, df: callable, x0: np.ndarray, lmbda: float, max_iter: int = 1000, tol: float = 1e-3, tol_grad_norm: float = 1e-4, verbose = False) -> np.ndarray:
    """Fast Iterative Soft Thresholding Algorithm
    Args:
        f: function to minimize
        df: gradient of the function
        x0: initial guess
        lmbda: regularization parameter
        max_iter: maximum number of iterations
        tol: tolerance for the stopping criterion
        tol_grad_norm: tolerance for the gradient norm

    Returns:
        solution of the optimization problem
    """

    gamma = 1
    all_x = [x0]

    cnt = 0
    while cnt < max_iter and distanceX > tol and grad_norm > tol_grad_norm:
        # compute the argument of the proximal operator
        y_current = y - gamma * df(y)

        # perform soft thresholding of x
        x_current = soft_thresholding(y_current, gamma * lmbda)

        # update alpha
        alpha_current = (1 + np.sqrt(1 + 4 * alpha ** 2)) / 2

        # update y
        y = x_current + ((alpha - 1) / alpha_current) * (x_current - x)

        # compute the stopping criteria

        distanceX = np.linalg.norm(x_current - x)
        grad_norm = np.linalg.norm(df(x_current))

        # store the estimate
        all_x.append(x_current)
        x = x_current
        alpha = alpha_current

        cnt += 1

        if verbose:
            print(f'Iteration {cnt}/{max_iter}\t Gradient norm: {grad_norm:.2f}\t Distance: {distanceX:.2f}')

    if verbose:
        print(f'Converged after {cnt} iterations')
        print(f'The minimum is {f(x):.2f}\t at x = {x}')

    return x


def weighted_LPA(s: np.ndarray, w: np.ndarray, N: int, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Weighted Local Polynomial Approximation
    Args:
        s: input signal
        w: weights
        N: degree of the polynomial
    Returns:
        smoothed signal
    """
    t = np.linspace(0, 1, M)
    T = np.column_stack([t**i for i in range(N+1)])

    w_inv = 1 / w
    w_inv = np.where(w_inv == np.inf, 0, w_inv)

    W = np.diag(w)
    W_inv = np.diag(w_inv)
    
    # comput the qr decomposition of WT
    # since T has more rows than columns, then qr computes only the first N + 1 columns of Q and the first N + 1 rows of R.
    Q, _ = np.linalg.qr(W @ T)

    #  define Qtilde
    Qtilde =  W_inv @ Q

    # adjust Qtilde with the weights matrix squared.
    W2Qtilde = W**2 @ Qtilde

    # select the central row of W2Qtilde
    row = M // 2

    # compute the kernel
    # g = np.sum(W2Qtilde[row, l] * W2Qtilde[:, l] for l in range(N + 1))
    g = Qtilde[row, :] @ W2Qtilde.T

    # flipping, since it is used in convolution
    g = g[::-1]

    # normalization
    g = g / np.sum(g)

    # convolution
    s_smooth = np.convolve(s, g, mode='same')

    return s_smooth, g

def compute_LPA_filter(w: np.ndarray, N: int) -> np.ndarray:
    """Compute the LPA filter
    Args:
        w: weights
        N: degree of the polynomial
    Returns:
        LPA filter
    """
    M = len(w)
    t = np.linspace(0, 1, M)
    T = np.column_stack([t**i for i in range(N+1)])

    w_inv = 1 / w
    w_inv = np.where(w_inv == np.inf, 0, w_inv)

    W = np.diag(w)
    W_inv = np.diag(w_inv)
    
    # comput the qr decomposition of WT
    # since T has more rows than columns, then qr computes only the first N + 1 columns of Q and the first N + 1 rows of R.
    Q, _ = np.linalg.qr(W @ T)

    #  define Qtilde
    Qtilde =  W_inv @ Q

    # adjust Qtilde with the weights matrix squared.
    W2Qtilde = W**2 @ Qtilde

    # select the central row of W2Qtilde
    row = M // 2

    # compute the kernel
    # g = np.sum(W2Qtilde[row, l] * W2Qtilde[:, l] for l in range(N + 1))
    g = Qtilde[row, :] @ W2Qtilde.T

    # flipping, since it is used in convolution
    g = g[::-1]

    # normalization
    g = g / np.sum(g)

    return g


def compute_2D_LPA_kernel(w, N):
    """Compute the 2D LPA kernel
    Args:
        w: weights
        N: degree of the polynomial
    Returns:
        2D LPA kernel
    """
    # window size is the lenght of the weight vector
    r, c = w.shape
    M = r * c
    
    # create the matrix T
    tx = np.linspace(0, 1, c)
    ty = np.linspace(0, 1, r)
    tx, ty = np.meshgrid(tx, ty)
    tx = tx.reshape(-1)
    ty = ty.reshape(-1)
    T = np.zeros((M,(N+1)**2))
    cnt = 0
    for i in range(N+1):
        for j in range(N-i+1):
            if i==0 and j==0:
                T[:, cnt] = np.ones(M)
            else:
                T[:, cnt] = tx**i * ty**j
            cnt = cnt + 1
    T = T[:, :cnt]

    # unroll the matrix of the weights    
    w = w.reshape(-1)

    # generate the inverse of weights
    winv = 1/w
    
    # set to zero weights that are inf
    winv = np.where(np.isinf(winv), 0, winv)

    # define the weight matrix
    W = np.diag(w)
    Winv = np.diag(winv)
    
    ## construct the LPA kernel
    
    # comput the qr decomposition of WT
    Q, _ = np.linalg.qr(W @ T)

    # define Qtilde
    Qtilde = Winv @ Q
    
    # adjust Qtilde with the weights matrix squared
    W2Qtilde = W**2 @ Qtilde

    # select the central row of W2Qtilde
    row = W2Qtilde[M // 2, :]

    # compute the kernel
    g_bar = np.sum([row[i] * Qtilde[:, i] for i in range(Qtilde.shape[1])], axis=0)

    #reshape the kernel in a matrix
    g_bar = g_bar.reshape(r, c)
    
    # flipping, since it is used in convolution
    g = g_bar[::-1, ::-1]

    return g


def fit_line_dlt(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    FIT_LINE_DLT - Fit a line to a set of points using the DLT algorithm
    Args:
        P: 2D points
    Returns:
        theta: parameters of the line
        residuals: residuals of the fit
        residual_error: sum of the squared residuals
    """

    # design matrix
    A = np.column_stack([P[:, 0], P[:, 1], np.ones_like(P[:, 0])])
    # vector of responses 
    y = P.T[:, 1]
    
    # SVD
    _, _, V_t = np.linalg.svd(A)

    theta = V_t[-1, :]
    
    residuals = A @ theta   
    residual_error = np.sum(residuals ** 2)
    
    return theta, residuals, residual_error

def IRLS(s: np.ndarray, D: np.ndarray, lmbda: float, x0: np.ndarray = None) -> np.ndarray:
    """Iteratively Reweighted Least Squares algorithm
    Args:
        s: input signal
        D: dictionary
        lmbda: regularization parameter
        x0: initial guess

    Returns:
        sparse representation of the input signal
        in the end it is minimizing the following objective function:
        min_x ||s - D x||_2^2 + lmbda ||x||_1
    """
    if x0 is None:
        x0 = np.ones(D.shape[1])

    delta = 1e-6
    max_iter = 20
    distanceX = 1e10
    tol_x = 1e-3
    all_x = [x0]
    cnt = 0

    while cnt < max_iter and distanceX > tol_x:
        x = all_x[-1]

        # compute the weight matrix
        W = np.diag(1 / (np.abs(x) + delta))

        # solve the weighted regularized LS system
        # x_current = np.linalg.inv((D.T @ D + 2 * lmbda * W)) @ (D.T @ s)

        x_current = np.linalg.solve((2 * lmbda * W + D.T @ D), D.T @ s)

        # update variable for stopping criteria (distance in x)
        distanceX = np.linalg.norm(x_current - x)

        # update all_x
        all_x.append(x_current)

        cnt += 1
    
    return all_x[-1]

def simpleRANSAC(X:np.ndarray, eps:float, cardmss:int, verbose = False) -> tuple[np.ndarray, np.ndarray, int]:
  """
  SIMPLERANSAC - Robust fit with the LMEDS algorithm
    Args:
        X: 2D points
        eps: threshold
        cardmss: minimal sample set size
    Returns:
        bestmodel: best model parameters
        bestinliers: inliers of the best model
        maxscore: number of inliers of the best model
  """
  # number of samples in the dataset
  n = X.shape[1] 
  # Desired probability of success
  alpha = 0.99 

  # Pessimistic estimate of inliers fraction
  f = 0.5

  # set maximum number of iterations
  MaxIterations = int(np.ceil(np.log(1-alpha) / np.log(1 - (1 - f)**cardmss)))

  # set maximum consensus reached
  maxscore = -np.inf

  bestinliers = []

  bestmodel = np.zeros((3, 1))
 
  for i in range(MaxIterations):
      
    # Generate cardmss random indices in the range 0..n-1
    mss = np.random.choice(n, cardmss, replace=False)
    
    # Fit model to this minimal sample set.
    theta = fit_line_dlt(X[:, mss].T)[0]

    # Evaluate distances between points and model
    sqres = res_line(X, theta)

    # identify inliers: consensus set
    inliers = np.where(sqres < eps)[0]

    # assess consensus (the number of inliers)
    score = len(inliers)

    # replace maxscore, bestinliers and bestmodel if needed
    bestinliers = inliers if score >= maxscore else bestinliers 
    bestmodel = theta if score >= maxscore else bestmodel
    maxscore = score if score > maxscore else maxscore

    if verbose:
        print(f'Iteration {i+1}/{MaxIterations}\t Inliers: {score}\t Best inliers: {maxscore}')

  return bestmodel, bestinliers, maxscore


def res_line(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    RES_LINE - Compute the residuals of a line model
    Args:
        X: 2D points
        M: line model
    Returns:
        d: residuals
    """
    if len(M.shape) > 1:
        num_lines = M.shape[1]
    else:
        num_lines = 1

    if num_lines == 1:
        d = np.abs(M[0] * X[0, :] + M[1] * X[1, :] + M[2])
    else:
        n = X.shape[1]
        d = np.zeros((n, num_lines))
        for i in range(num_lines):
            d[:, i] = np.abs(M[0, i] * X[0, :] + M[1, i] * X[1, :] + M[2, i])

    return d

def fit_circle(P: np.ndarray) -> tuple[np.ndarray, float]:
    """
    FIT_CIRCLE - Fit a circle to a set of points
    Args:
        P: 2D points (3 points are needed for the circle to be uniquely determined)
    Returns:
        center: (x and y coordinates)
        radius: radius of the circle
    """
    A = np.hstack((P.T, np.ones((3, 1))))
    B = -np.sum(P**2, axis=0).reshape(-1, 1)

    D, E, F = np.linalg.solve(A, B).flatten()
    x, y = -D / 2, -E / 2
    r = np.sqrt(x**2 + y**2 - F)

    return [x, y], r

def res_circle(X: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    """
    RES_CIRCLE - Compute the residuals of a circle model
    Args:
        X: 2D points
        center: center of the circle
        radius: radius of the circle
    Returns:
        d: residuals
    """
    x, y = center

    # Compute the distance from each point to the circle
    return np.sqrt((X[0, :] - x) ** 2 + (X[1, :] - y) ** 2) - radius

def circlesRANSAC(X:np.ndarray, eps:float, cardmss:int = 3, verbose = False) -> tuple[np.ndarray, np.ndarray, int]:
  """
  RANSAC - Fit circles to a set of points using the RANSAC algorithm
    Args:
        X: 2D points
        eps: threshold
        cardmss: minimal sample set size
    Returns:
        bestmodel: best model parameters
        bestinliers: inliers of the best model
        maxscore: number of inliers of the best model
  """
  # number of samples in the dataset
  n = X.shape[1] 
  # Desired probability of success
  alpha = 0.99 

  # Pessimistic estimate of inliers fraction
  f = 0.75

  # set maximum number of iterations
  MaxIterations = int(np.ceil(np.log(1-alpha) / np.log(1 - (1 - f)**cardmss)))

  # set maximum consensus reached
  maxscore = -np.inf

  bestinliers = []

  bestmodel = (np.zeros((2, 1)), 0)
 
  for i in range(MaxIterations):
      
    # Generate cardmss random indices in the range 0..n-1
    mss = np.random.choice(n, cardmss, replace=False)
    
    # Fit model to this minimal sample set.
    center, radius = fit_circle(X[:, mss])

    # Evaluate distances between points and model
    sqres = res_circle(X, center, radius)

    # identify inliers: consensus set
    inliers = np.where(np.abs(sqres) < eps)[0]

    # assess consensus (the number of inliers)
    score = len(inliers)

    # replace maxscore, bestinliers and bestmodel if needed
    bestinliers = inliers if score >= maxscore else bestinliers 
    bestmodel = (center, radius) if score >= maxscore else bestmodel
    maxscore = score if score > maxscore else maxscore

    if verbose:
        print(f'Iteration {i+1}/{MaxIterations}\t Inliers: {score}\t Best inliers: {maxscore}')

  return bestmodel, bestinliers, maxscore