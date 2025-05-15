
import getopt
import sys
from tensap.poincare_learning.benchmarks.poincare_benchmarks_torch import build_benchmark_torch
from tensap.poincare_learning.utils._sklearn_wrapper import PolynomialFeatureEstimator
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

DEBUG = True

def get_params():
    """
    Parses command-line arguments converts them into a dictionary of key-value pairs. 
    It supports three types of parameters: strings (-s), booleans (-b), and integers (-i).
    The command-line options should be of the form key1:value1+key2:value2, for example `-s name:John+city:NewYork -b verbose:true+debug:false -i age:25+limit:50`.
    This function is only used when runing this python file from command line.

    Returns
    -------
    key : dict
        Dictionary where keys are parameter names and values are converted to appropriate types (string, boolean, or integer).
    """
    optlist, _ = getopt.getopt(sys.argv[1:], 'i:s:b:')
    key = {}
    for i in optlist:
        print(i)
        if i[0] == '-s':
            for a in i[1].split(sep='+'):
                k, v = a.split(sep=':')
                key[k] = v
        elif i[0] == '-b':
            for a in i[1].split(sep='+'):
                k, v = a.split(sep=':')
                key[k] = True if v == "true" else False
        elif i[0] == '-i':
            for a in i[1].split(sep='+'):
                k, v = a.split(sep=':')
                key[k] = int(v)
        else:
            print('error:', i[0], i[1])

    return key


def build_benchmark_name_kwargs(params):
    """
    Converts the parsed parameters into the corresponding name and kwargs dictionary parsable to `build_benchmark_torch`, which build the desired benchmark with desired parameters.
    
    Parameters
    ----------
    params : dict
        Dictionary of raw parameters, coming for example from `get_params`.
    Returns
    -------
    benchmark_name : str
        Tensap name of the benchmark.
    benchmark_kwargs : dict
        Key word arguments used to build the Tensap benchmark.

    """

    benchmark_name = params.get('benchname')
    benchmark_kwargs = {}
    if benchmark_name == 'borehole':
        benchmark_kwargs = {}

    elif benchmark_name in ('sin_squared_norm', 'exp_mean_sin_exp_cos', 'sum_cos_sin_squared_norm'):
        d = params.get('d')
        benchmark_kwargs['d'] = d
        benchmark_kwargs['c'] = np.pi / 2

        if benchmark_name == "sum_cos_sin_squared_norm":
            benchmark_kwargs['R1'] = np.eye(d) / 2
            benchmark_kwargs['R2'] = np.array([[1 / (1+i+j) for i in range(d)] for j in range(d)]) / 2

    else:
        raise NotImplementedError('Benchmark name not implemented')

    return benchmark_name, benchmark_kwargs


def build_estimator_features(params):
    """
    Create a PolynomialFeatureEstimator from parsed parameters.

    Parameters
    ----------
    params : dict
        Dictionary of raw parameters, coming for example from `get_params`.
    Returns
    -------
    estimator : PolynomialFeatureEstimator
        Sklearn estimator with parsed parameters

    """

    fit_method = params.get('fitmethod')

    if DEBUG: verbosity = 2
    else: verbosity = 1

    if fit_method == 'surgreedy':
        fit_method = "surrogate_greedy"

    # init method when using pymanopt minimizer
    init_method = params.get('initmethod')
    if init_method == 'surgreedy':
        init_method = "surrogate_greedy"

    elif init_method == 'randlin':
        init_method = "random_linear"

    elif init_method == 'as':
        init_method = 'active_subspace'    

    if fit_method == 'pymanopt':
        
        # kwargs when using pymanopt minimizer 
        optimizer_kwargs = {
                'beta_rule': 'PolakRibiere',
                'min_step_size': 1e-12,
                'max_iterations': 200, 
                'verbosity': verbosity,
                'log_verbosity':1
            }

        fit_parameters = {
            "m": params.get('m'),
            "init_method": init_method,
            "precond_kwargs":{}, 
            "optimizer_kwargs":optimizer_kwargs, 
            "ls_kwargs":{},
        }  

    elif fit_method == 'surrogate':

        fit_parameters = {
            "m": params.get('m')
            }
    
    elif fit_method == 'surrogate_greedy':

        optimizer_kwargs = {
                'beta_rule': 'PolakRibiere',
                'min_step_size': 1e-12,
                'max_iterations': 200, 
                'verbosity': verbosity,
                'log_verbosity':1
            }

        pmo_kwargs = {
            'use_precond':True, 
            'precond_kwargs':{}, 
            'optimizer_kwargs':optimizer_kwargs, 
            'ls_kwargs':{}
        }

        fit_parameters =  {
            "m_max": params.get("m"),
            "optimize_poincare": params.get("opti"),
            "tol":1e-15,
            "pmo_kwargs": pmo_kwargs,
        }
        
    else:
        raise NotImplementedError("fit_method not valid")

    if DEBUG and not(optimizer_kwargs.get('max_iterations') is None):
        optimizer_kwargs['max_iterations'] #//=20


    benchmark_name, benchmark_kwargs = build_benchmark_name_kwargs(params)

    _, _, rand_vec = build_benchmark_torch(benchmark_name, **benchmark_kwargs)
    
    estimator = PolynomialFeatureEstimator(rand_vec, p_norm=1, max_p_norm=1, innerp='h1_0', fit_method=fit_method, fit_parameters=fit_parameters)

    return estimator


def build_samples(params, seed):
    """
    Create train and test samples for x, u(x) and jac_u(x), where u is the desired benchmark.

    Parameters
    ----------
    params : dict
        Dictionary of raw parameters, coming for example from `get_params`.
    seed : int
        Seed for the random number generator.
    """

    # build the benchmark
    benchmark_name, benchmark_kwargs = build_benchmark_name_kwargs(params)
    u, jac_u, rand_vec = build_benchmark_torch(benchmark_name, **benchmark_kwargs)

    # get sample sizes
    n_train = script_params.get('ntrain')
    n_test = script_params.get('ntest')

    # initialize the RNG for reproductibility
    logging.info(f"Initializing numpy random Generator with seed {seed}")

    # generate the samples with lhs
    logging.info(f"Generating sample of X, u, jac_u of size with n_train={n_train} and n_test={n_test}")

    logging.info(f"Generating samples of X")
    x_train = rand_vec.lhs_random(n_train, seed=seed)
    np.random.RandomState(seed+1).shuffle(x_train)
    x_test = rand_vec.lhs_random(n_test, seed=seed+2**15)

    logging.info(f"Generating samples of u(X)")
    u_train = u(x_train)
    u_test = u(x_test)

    logging.info(f"Generating samples of jac_u(X)")
    jac_u_train = jac_u(x_train)
    jac_u_test = jac_u(x_test)

    return x_train, u_train, jac_u_train, x_test, u_test, jac_u_test


def build_cv_features(params):
    """
    Create a model selector for fitting the hyperparameters of the polynomial feature map.
    For more details on the latter, see A. Nouy and A. Pasco 2025.

    Parameters
    ----------
    params : dict
        Dictionary of raw parameters, coming for example from `get_params`.
    Returns
    -------
    cv_features : GridSearchCV
        Sklearn model slector for the polynomial feature map.
    """
    estimator = build_estimator_features(params)

    # Parameter grids for CV of features
    param_grid_0 = {
        'p_norm':[0.8],
        'max_p_norm':[2,3,4,5],
        'innerp':['h1_0'],
    }

    param_grid_1 = {
        'p_norm':[0.9],
        'max_p_norm':[2,3,4],
        'innerp':['h1_0'],
    }

    param_grid_2 = {
        'p_norm':[1.],
        'max_p_norm':[1,2,3],
        'innerp':['h1_0'],
    }

    param_grid = [param_grid_0, param_grid_1, param_grid_2]

    if DEBUG:
        param_grid = param_grid[0]
        param_grid['max_p_norm'] = [5]

    # build the CV model
    cv_features = GridSearchCV(
        estimator,
        param_grid=param_grid,
        verbose=4,
    )

    return cv_features


def build_cv_profile(params):
    """
    Create a model selector for fitting the hyperparameters of the profile map.
    For more details on the latter, see A. Nouy and A. Pasco 2025.

    Parameters
    ----------
    params : dict
        Dictionary of raw parameters, coming for example from `get_params`.
        Actually, not used.
    Returns
    -------
    cv_profile : GridSearchCV
        Sklearn model slector for the profile map.
    """
    krr_param_grid = {
                'alpha': np.logspace(-11, -5, 40),
                'gamma': np.logspace(-6, -2, 30),
            }
    
    if DEBUG:
        krr_param_grid = {
                'alpha': np.logspace(-10, -4, 5),
                'gamma': np.logspace(-6, -3, 5),
            }

    cv_profile = GridSearchCV(
            KernelRidge(kernel="rbf"),
            param_grid=krr_param_grid,
            scoring="neg_mean_squared_error",
            verbose=4,
            cv=10
        )
    
    return cv_profile

def unit_run(params, seed):
    """
    Run the full learning procedure from a single dataset generated with parsed seed and params.
    For more details on the latter, see A. Nouy and A. Pasco 2025.

    Parameters
    ----------
    params : dict
        Dictionary of raw parameters, coming for example from `get_params`.
    seed : int
        Seed for the random number generator.

    Returns
    -------
    results : dict
        Contains detailed results of the run.
    """
    results = dict()
    results['script_params'] = params
    log_prefix = f"Seed{seed}:"
    # generate the samples

    x_train, u_train, jac_u_train, x_test, u_test, jac_u_test = build_samples(params, seed)

    results['x_train'] = x_train
    results['x_test'] = x_test
    results['u_train'] = u_train
    results['u_test'] = u_test
    results['jac_u_train'] = jac_u_train
    results['jac_u_test'] = jac_u_test

    # fit the features using CV
    
    cv_features = build_cv_features(params)
    logging.info(log_prefix + f"Features:Cross Validation")
    cv_features.fit(x_train, jac_u_train)
    logging.info(log_prefix + f"Features:Best parameters {cv_features.best_params_} with neg Poincare loss {cv_features.best_score_:.3e}")

    results['cv_features'] = cv_features
    results['optim_history'] = cv_features.best_estimator_.optim_history

    # evaluate performances on train and test set

    ploss_train = -cv_features.score(x_train, jac_u_train)
    ploss_test = -cv_features.score(x_test, jac_u_test)

    results['fitted_ploss_train'] = ploss_train
    results['fitted_ploss_test'] = ploss_test

    logging.info(log_prefix + f"Features:Fitted train Poincare loss: {ploss_train:.3e}")
    logging.info(log_prefix + f"Features:Fitted test Poincare loss: {ploss_test:.3e}")

    # generate the features

    z_train = cv_features.best_estimator_.g.eval(x_train)
    z_test = cv_features.best_estimator_.g.eval(x_test) 
    results['z_train'] = z_train
    results['z_test'] = z_test
    
    # fit the regressor using kernel ridge

    cv_profile = build_cv_profile(params)
    logging.info(log_prefix + f"Profile:Cross Validation")
    cv_profile.fit(z_train, u_train)
    logging.info(log_prefix + f"Profile:best parameters {cv_profile.best_params_} with neg mse {cv_profile.best_score_:.3e}")

    results['cv_profile'] = cv_profile

    # evaluate performances on train and test set
    mse_train = -1. * cv_profile.score(z_train, u_train)
    mse_test = -1. * cv_profile.score(z_test, u_test)  

    results['fitted_mse_train'] = mse_train
    results['fitted_mse_test'] = mse_test

    logging.info(log_prefix + f"Profile:Fitted train MSE: {mse_train:.3e}")
    logging.info(log_prefix + f"Profile:Fitted test MSE: {mse_test:.3e}")

    # save the results
    save_name = params.get('diroutput') + "results_seed_" + str(seed) + ".npy"
    np.save(save_name, results)
    logging.info(log_prefix + f"Detailed results saved at {save_name}")
    
    return results


if __name__ == "__main__":

    # %% get the parameters from the bash script

    if DEBUG :
        script_params = {
            'benchname':'sum_cos_sin_squared_norm',
            'nseeds':2,
            'fitmethod':'pymanopt',
            'diroutput':'./tensap/numerics/debug/',
            'd':8,
            'm':1,
            'initmethod': 'surrogate',
            'opti':False,
            'ntrain':100,
            'ntest':1000
        }

    else:
        script_params = get_params()

    
    # %% iterate over all rng seeds
    results_all_seeds = []

    for rngseed in range(script_params.get('nseeds')):
        results = unit_run(script_params, rngseed)
        results_all_seeds.append(results)

    # %% save the results with all seeds
    save_name = script_params.get('diroutput') + "results_all_seeds.npy"
    np.save(save_name, results)
    logging.info(f"Detailed results saved at {save_name}")

