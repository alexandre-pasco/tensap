

# %% Imports
import numpy as np
import matplotlib.pyplot as plt
from tensap.poincare_learning.benchmarks.poincare_benchmarks import build_benchmark
from tensap.poincare_learning.utils._loss_vector_space import _build_ortho_poly_basis
from tensap.poincare_learning.poincare_loss_vector_space import PoincareLossVectorSpace, PoincareLossVectorSpaceTruncated
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import logging
logging.basicConfig(level=logging.INFO)

# %% Function to generate samples

def generate_samples(N, ind1, X, fun, jac_fun, basis, R=None):
    x_set = X.lhs_random(N)
    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)[:, ind1]
    basis_set = basis.eval(x_set[:, ind1])
    jac_basis_set = basis.eval_jacobian(x_set[:, ind1])

    if fun_set.ndim == 1:
        fun_set = fun_set[:, None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:, None, :]

    loss_set = PoincareLossVectorSpace(jac_fun_set, jac_basis_set, basis, R)

    return x_set, fun_set, jac_fun_set, basis_set, jac_basis_set, loss_set

def generate_samples_tensorized(N1, N2, ind1, X, fun, jac_fun, basis, R=None):

    ind2 = np.delete(np.arange(X.ndim()), ind1)

    X1 = X.marginal(ind1)
    X2 = X.marginal(ind2)

    x1_set = X1.lhs_random(N1)
    x2_set = X2.lhs_random(N2)
    
    x_set = np.zeros((N1*N2, X.ndim()))
    jac_basis_set = basis.eval_jacobian(x1_set)
    jac_fun_set_tensorized = np.zeros((N1, N2, len(ind1)))

    for i in range(N1):
        # x2_set = X2.lhs_random(N2)
        x_set_i = np.zeros((N2 ,X.ndim()))
        x_set_i[:, ind1] = x1_set[i]
        x_set_i[:, ind2] = x2_set
        x_set[i*N2:(i+1)*N2] = x_set_i
        jac_fun_set_tensorized[i,:,:] = jac_fun(x_set_i)[:,ind1]

    fun_set = fun(x_set)
    jac_fun_set = jac_fun(x_set)
    basis_set = basis.eval(x1_set)
    jac_basis_set = basis.eval_jacobian(x1_set)

    if fun_set.ndim == 1:
        fun_set = fun_set[:, None]

    if jac_fun_set.ndim == 2:
        jac_fun_set = jac_fun_set[:, None, :]

    loss_set_tensorized = PoincareLossVectorSpaceTruncated(jac_fun_set_tensorized, jac_basis_set, basis, R)

    return x_set, fun_set, jac_fun_set_tensorized, basis_set, jac_basis_set, loss_set_tensorized


# %% Functions to fit regressors

def fit_krr_regressor(z_set, u_set):

    kr = GridSearchCV(
        KernelRidge(kernel="rbf", gamma=0.1),
        param_grid={
            "alpha": np.logspace(-11, -5, 10),
            "gamma": np.logspace(-6, 2, 10)},
        scoring="neg_mean_squared_error"
    )

    kr.fit(z_set, u_set)
    print(f"Best KRR with params: {kr.best_params_} and MSE score: {-kr.best_score_:.3e}")

    return kr


def fit_poly_regressor(z_set, u_set):
    model = Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression(fit_intercept=False))
    ])

    param_grid = {
        'poly__degree': np.arange(10)
    }

    cv = GridSearchCV(
        model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=10
    )

    cv.fit(z_set, u_set)
    print(f"Best Poly with params: {cv.best_params_} and  score: {-cv.best_score_:.3e}")

    return cv


# %% functions to build matrice list

def build_mat_lst_1(dim, n_mat):
    mat_lst = [np.eye(dim)]
    for i in range(1, n_mat):
        mat = np.eye(dim)
        for j in range(1, i+1):
            mat += np.diag(np.ones(dim-j), k=j)
            mat += np.diag(np.ones(dim-j), k=-j)
        mat = mat / i
        mat_lst.append(mat)
    return mat_lst

def build_mat_lst_2(dim, n_mat):
    mat_lst = [np.eye(dim)]
    for i in range(1, n_mat):
        mat = np.diag(np.ones(dim-i), k=i)
        mat = mat + mat.T
        mat = mat / 2
        mat_lst.append(mat)
    return mat_lst

def build_mat_lst_3(dim, n_mat):
    mat_lst = []
    for i in range(n_mat):
        mat = np.random.RandomState(i).normal(size=(dim, dim))
        mat = mat.T @ mat
        mat = mat / np.linalg.norm(mat, ord=2)
        mat_lst.append(mat)
    return mat_lst

def build_mat_lst(dim, n_mat, which=1):
    if which == 1:
        out = build_mat_lst_1(dim, n_mat)
    elif which == 2:
        out = build_mat_lst_2(dim, n_mat)
    elif which == 3:
        out = build_mat_lst_3(dim, n_mat)
    else:
        raise NotImplementedError
    return out

# %% Definition of the benchmark

# build list of matrices 
d = 16
n_mat = 2
ind1 = np.arange(d//2)
ind2 = np.delete(np.arange(d), ind1)
mat_lst_1 = build_mat_lst(d//2, n_mat, which=2)
mat_lst_2 = build_mat_lst(d//2, n_mat, which=2)


# if pytorch is installed
try:
    from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import build_benchmark_torch
    u, jac_u, X = build_benchmark_torch("quartic_sin_two_variables", d=d, mat_lst_1=mat_lst_1, mat_lst_2=mat_lst_2)
    

except ImportError:
    u, jac_u, X = build_benchmark("quartic_sin_collective", d=d, mat_lst=mat_lst_1)

# %% build a polynomial basis

p_norm = 1  # p-norm of the multi-indices
max_deg = 2  # bound on the p_norm
basis1 = _build_ortho_poly_basis(X.marginal(ind1), p=p_norm, m=max_deg)
basis2 = _build_ortho_poly_basis(X.marginal(ind2), p=p_norm, m=max_deg)
K1 = basis1.cardinal()
R1 = basis1.gram_matrix_h1_0()
K2 = basis2.cardinal()
R2 = basis2.gram_matrix_h1_0()

#####################################
# %% Feature first group with surrogate
#####################################

# %% Sampling

N1_train = 20
N2_train = 20
x_train, u_train, _, _, _, loss_train_tensorized_1 = generate_samples_tensorized(N1_train, N2_train, ind1, X, u, jac_u, basis1, R1)

N_test = 1000
x_test, u_test, _, _, _, loss_test_1 = generate_samples(
    N_test, ind1, X, u, jac_u, basis1, R1)

# %% Minimize the surrogate

m = len(mat_lst_1) - 0 
loss_train_tensorized_1.truncate(m)
G1_surr, _, _ = loss_train_tensorized_1.minimize_surrogate(m=m)
# G1_surr, _, _ = loss_train_tensorized.minimize_pymanopt(G0=G1_surr, optimizer_kwargs={'max_iterations':100})
G1_surr = np.linalg.svd(G1_surr, full_matrices=False)[0]

# %% Evaluate performances

print(f"\nPoincare loss and Surrogate on {G1_surr.shape[1]} features")
print(f"Surrogate on tensorized train set:      {loss_train_tensorized_1.eval_surrogate(G1_surr):.3e}")
print(f"Poincare loss on tensorized train set : {loss_train_tensorized_1.eval(G1_surr):.3e}")
print(f"Poincare loss on test set:              {loss_test_1.eval(G1_surr):.3e}")

# %% Plot for eyeball regression
def plot_eyeball(x_set, ind, basis, G, N1, N2):
    z_set = basis.eval(x_set[:, ind]) @ G

    fig, ax = plt.subplots(1, z_set.shape[1])
    if z_set.shape[1] == 1:
        ax = [ax]
    ax[0].set_ylabel('u(X)')

    for i in range(z_set.shape[1]):
        for j in range(N1):
            ax[i].scatter(z_set[j::N2, i], u_train[j::N2], label='train', s=5.)
        ax[i].set_xlabel(f'g_{i}(X)')

    fig.suptitle(f"""
        Surrogate only tensorized sample
        Poly features m={z_set.shape[1]}
        Multi-indices with {p_norm}-norm bounded by {max_deg}
        {N1}x{N2}={N1*N2} train samples
        """,
        y=0.)
    plt.show()

# %%
plot_eyeball(x_train, ind1, basis1, G1_surr, N1_train, N2_train)


#####################################
# %% Feature second group with surrogate
#####################################

# %% Sampling

N1_train = 20
N2_train = 20
x_train, u_train, _, _, _, loss_train_tensorized_2 = generate_samples_tensorized(N2_train, N1_train, ind2, X, u, jac_u, basis2, R2)

x_test, u_test, _, _, _, loss_test_2 = generate_samples(
    N_test, ind2, X, u, jac_u, basis2, R2)

# %% Minimize the surrogate

m = len(mat_lst_2) - 0 
loss_train_tensorized_2.truncate(m)
G2_surr, _, _ = loss_train_tensorized_2.minimize_surrogate(m=m)
G2_surr = np.linalg.svd(G2_surr, full_matrices=False)[0]

# %% Evaluate performances

print(f"\nPoincare loss and Surrogate on {G2_surr.shape[1]} features")
print(f"Surrogate on tensorized train set:      {loss_train_tensorized_2.eval_surrogate(G2_surr):.3e}")
print(f"Poincare loss on tensorized train set : {loss_train_tensorized_2.eval(G2_surr):.3e}")
print(f"Poincare loss on test set:              {loss_test_2.eval(G2_surr):.3e}")

# %% Plot for eyeball regression

plot_eyeball(x_train, ind2, basis2, G2_surr, N2_train, N1_train)


#####################################
# %% Fit final regressor
#####################################

# %% Fit regressor with sklearn

def fit_regressor(x_set, ind1, ind2, basis1, basis2, G1, G2, u_set):

    z1_set = basis1.eval(x_set[:, ind1]) @ G1
    z2_set = basis2.eval(x_set[:, ind2]) @ G2

    # add the last parameter as a feature
    z_set = np.hstack([z1_set, z2_set])
    
    regressor = fit_krr_regressor(z_set, u_set)
    # regressor = fit_poly_regressor(z_set, u_set)

    def profil(z1, z2):
        return regressor.predict(np.hstack([z1, z2]))
    
    return profil

# %%

f_surr = fit_regressor(x_train, ind1, ind2, basis1, basis2, G1_surr, G2_surr, u_train)

# %% Evaluate performances

def eval_perf(G1, G2):
    z1_train = basis1.eval(x_train[:, ind1]) @ G1
    z2_train = basis2.eval(x_train[:, ind2]) @ G2

    y_train = f_surr(z1_train, z2_train).reshape(u_train.shape)
    err_train = np.sqrt(np.mean((y_train - u_train)**2))
    rel_err_train = err_train / np.sqrt((u_train**2).mean())

    z1_test = basis1.eval(x_test[:, ind1]) @ G1
    z2_test = basis2.eval(x_test[:, ind2]) @ G2

    y_test = f_surr(z1_test, z2_test).reshape(u_test.shape)
    err_test = np.sqrt(np.mean((y_test - u_test)**2))
    rel_err_test = err_test / np.sqrt((u_test**2).mean())

    print(f"\nSurrogate only tensorized sample | Regression based on {G1_surr.shape[1]} features")
    print(f"L2 on train set    : {err_train:.3e}")
    print(f"L2 on test set     : {err_test:.3e}")
    print(f"RL2 on train set   : {rel_err_train:.3e}")
    print(f"RL2 on test set    : {rel_err_test:.3e}")

    plt.scatter(y_test, u_test, label='test', s=5.)
    plt.scatter(y_train, u_train, label='train', s=5.)
    plt.ylabel("u(X)")
    plt.xlabel("f(g(X))")
    plt.legend()
    plt.title(f"""
        Surrogate only tensorized sample
        Poly features m={z1_train.shape[1] + z2_train.shape[1]}
        Multi-indices with {p_norm}-norm bounded by {max_deg}
        {N1_train}x{N2_train}={N1_train*N2_train} train samples""")
    plt.show()

# %%
eval_perf(G1_surr, G2_surr)

