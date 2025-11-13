

# %% import libraries
import numpy as np
import matplotlib.pyplot as plt


# %% Some variables
#n_train_lst = [50, 75, 100, 125, 150, 250, 500]
n_train_lst = [75]
nseeds = 20

# def function to concatenate
def concatenate_all_seeds(path):

    # load the results
    results = []
    for i in range(nseeds):
        name = path + "results_seed_" + str(i) + ".npy"
        res_i = np.load(name, allow_pickle=True).item()
        results.append(res_i)
    
    # concatenate for each keys
    concatenated_results = {}
    for key in results[0].keys():
        concatenated_results[key] = np.array([r[key] for r in results])
    
    return concatenated_results

def draw_fitted_boxplots(path):
    suffix_lst=[
        "fitmethod_surrogate_ntrain_75_nytrain_5_ntest_1000",
    ]

    tik_names = [
        "sur",
        #"pmo as",
        #"pmo sur",
    ]

    plotted_keys = [
        'fitted_mse_test',
        'fitted_mse_train',
        'fitted_ploss_test',
        'fitted_ploss_train',
    ]
    
    results = [concatenate_all_seeds(path + suf) for suf in suffix_lst]

    fig, ax = plt.subplots(len(plotted_keys), figsize=(10,15))
    for i, key in enumerate(plotted_keys):
        ax[i].boxplot([r.get(key) for r in results], tick_labels=tik_names)
        ax[i].set_title(key)
        ax[i].set_yscale('log')


def draw_fitted_vs_ntrain(path):
    prefix_lst=[
        "fitmethod_surrogate_ntrain_",
        #"fitmethod_pymanopt_init_as_d_8_m_1_ntrain_",
        #"fitmethod_pymanopt_init_surrogate_d_8_m_1_ntrain_",
    ]

    suffix_lst = [str(n) + "_nytrain_5_ntest_1000/" for n in n_train_lst]

    labels = [
        "sur",
        #"pmo as",
        #"pmo sur",
    ]

    plotted_keys = [
        'fitted_mse_test',
        'fitted_mse_train',
        'fitted_ploss_test',
        'fitted_ploss_train',
    ]


    fig, ax = plt.subplots(len(plotted_keys),1, figsize=(10,15))

    colors = ['r','g','b']
    for i, prefix in enumerate(prefix_lst):
        results = [concatenate_all_seeds(path + prefix + suffix) for suffix in suffix_lst]

        for j, key in enumerate(plotted_keys):
            vals = np.array([r.get(key).mean() for r in results])
            q0 = np.array([np.quantile(r.get(key), 0) for r in results])
            q20 = np.array([np.quantile(r.get(key), 0.2) for r in results])
            q80 = np.array([np.quantile(r.get(key), 0.8) for r in results])
            q100 = np.array([np.quantile(r.get(key), 1) for r in results])
            ax[j].plot(n_train_lst, vals, label=labels[i], color=colors[i])
            ax[j].fill_between(n_train_lst, q0, q100, alpha=0.1, color=colors[i])
            ax[j].fill_between(n_train_lst, q20, q80, alpha=0.5, color=colors[i])

    for j, key in enumerate(plotted_keys):
        ax[j].set_xticks(n_train_lst)
        ax[j].set_yscale('log')
        ax[j].set_xscale('log')
        ax[j].legend()
        ax[j].set_title(key)

    plt.show()


def draw_fit_history(path):

    prefix_lst=[
        "fitmethod_pymanopt_init_as_d_8_m_1_ntrain_",
        "fitmethod_pymanopt_init_surrogate_d_8_m_1_ntrain_",
    ]

    labels = [
        "pmo as",
        "pmo sur",
    ]

    fig, ax = plt.subplots(len(n_train_lst),1, figsize=(10,15))

    for j, key in enumerate(n_train_lst):
        for i, prefix in enumerate(prefix_lst):
            name = path + prefix + f"{key}_ntest_1000/"
            results = concatenate_all_seeds(name)
            color = ['r', 'g', 'b', 'c', 'b'][i]
            for opti_hist in results['optim_history']:
                costs = opti_hist['cost']
                ax[j].plot(costs, c=color)

    for j, key in enumerate(n_train_lst):
        ax[j].set_xscale('log')
        ax[j].set_yscale('log')
        ax[j].legend(labels=labels, labelcolor=['r','g'], handlelength=0)
        ax[j].set_title(key)

    plt.show()


def draw_cost_hist_quant(path):

    prefix_lst=[
        "fitmethod_pymanopt_ntrain_75_nytrain_5_ntest_1000"
    ]

    labels = [
        "pmo as",
        "pmo sur",
    ]
    
    quantiles = [0., 0.1, 0.9, 1.]

    fig, ax = plt.subplots(len(n_train_lst),1, figsize=(10,15))

    for j, n_train in enumerate(n_train_lst):
        for i, prefix in enumerate(prefix_lst):
            color = ['r', 'g', 'b', 'c', 'b'][i]
            name = path + prefix + f"{n_train}_ntest_1000/"
            res = concatenate_all_seeds(name) 
            cost_hist = np.zeros((600, len(res['script_params'])))
    
            # repeat last value when needed
            for k, oph in enumerate(res['optim_history']):
                cost_hist[:, k] = np.concatenate([oph['cost'], oph['cost'][-1] * np.ones(600 - len(oph['cost']))])
            
            cost_quant_hist = np.quantile(cost_hist, quantiles, axis=1)
            for cost_q in cost_quant_hist:
                ax[j].plot(cost_q, c=color)

    for j, key in enumerate(n_train_lst):
        ax[j].set_yscale('log')
        ax[j].legend(labels=labels, labelcolor=['r','g'], handlelength=0)
        ax[j].set_title(key)


# %%
results_path = "/home/apasco/Documents/python/tensap/numerics/surrogate_collective_poincare/outputs/quartic_sin_collective_d_7_m_3_nmat_3/"
#results_path = "/home/apasco/Documents/python/tensap/numerics/outputs/exp_mean_sin_exp_cos/"

# %%
draw_fitted_vs_ntrain(results_path)

# %%
#draw_fitted_boxplots(results_path)

# %%
#draw_fit_history(results_path)





# %%
path = results_path

# %%
name="fitmethod_surrogate_ntrain_75_nytrain_5_ntest_1000/"
#name="fitmethod_pymanopt_init_surrogate_d_8_m_1_ntrain_50_ntest_1000/" 
results = concatenate_all_seeds(path + name) 

#ind = results['fitted_mse_test'].argmax()
ind = np.argmin([results['optim_history'][i]['cost'].min() for i in range(16)])

# %%
cv_features = results['cv_features'][ind]
cv_profile = results['cv_profile'][ind]

# %%
z_train = results['z_train'][ind]
u_train = results['u_train'][ind]

z_test= results['z_test'][ind]
u_test = results['u_test'][ind]

# %%
plt.scatter(z_train, u_train, label='train')
plt.scatter(z_test, u_test, label='test')
plt.legend()
plt.show()

# %%
plt.scatter(cv_profile.predict(z_train), u_train, label='train')
plt.scatter(cv_profile.predict(z_test), u_test, label='test')
plt.legend()
plt.show()


# %%
