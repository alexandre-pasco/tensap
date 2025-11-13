

# %% import libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import os
import logging


root_path = "./"
n_train_lst = [50, 75, 100, 125, 150, 250]
quantiles = [0., 0.1, 0.2, 0.5, 0.8, 0.9, 1.]


# def function to concatenate
def concatenate_all_seeds(path):
    logging.debug(f"concatenating_all_seeds for {path}")
    nseeds = 20

    # load the results
    results = []
    for i in range(nseeds):
        name = path + "results_seed_" + str(i) + ".npy"
        res_i = np.load(name, allow_pickle=True).item()
        results.append(res_i)
    
    # concatenate for each keys
    concatenated_results = {}
    for key in results[0].keys():
        concatenated_results[key] = [r[key] for r in results]
    
    # add zeros to features if necessary
    m = np.max([r['z_train'].shape[1] for r in results])
    for i in range(nseeds):
        for key in ['z_train', 'z_test']:
            n, m_eff = concatenated_results[key][i].shape
            if m_eff < m:
                concatenated_results[key][i] = np.hstack([concatenated_results[key][i], np.zeros((n, m-m_eff))])
    
    # transform everything into numpy arrays
    for key in results[0].keys():
        for i in range(nseeds):
            concatenated_results[key] = np.array(concatenated_results[key])

    return concatenated_results


def get_prefix_label_lst(benchname):

    prefix_lst=[
        "fitmethod_surrogate_ntrain_",
        #"fitmethod_surgreedy_opti_false_ntrain_",
        #"fitmethod_surgreedy_opti_true_ntrain_",
        "fitmethod_pymanopt_init_as_ntrain_",
        #"fitmethod_pymanopt_init_surrogate_ntrain_",
        #"fitmethod_pymanopt_init_surgreedy_ntrain_",
    ]
    labels = [
        "sur",
        #"surgreedy",
        #"surgreedy opti",
        "pmo as",
        #"pmo sur",
        #"pmo surgreedy",
    ]
    prefix_cost_hist_lst=[
        "fitmethod_pymanopt_init_as_ntrain_",
        #"fitmethod_pymanopt_init_surrogate_ntrain_",
        #"fitmethod_pymanopt_init_surgreedy_ntrain_",
    ]
    labels_cost_hist = [
        "pmo as",
        #"pmo sur",
        #"pmo surgreedy",
    ]

    return prefix_lst, prefix_cost_hist_lst, labels, labels_cost_hist


def get_fitted_vs_ntrain(path):

    key_lst = [
        ('fitted_mse_test', 'rl2_test'),
        ('fitted_mse_train', 'rl2_train'),
        ('fitted_ploss_test', 'rsqrtploss_test'),
        ('fitted_ploss_train', 'rsqrtploss_train'),
    ]

    header = "n_train"
    for keys in key_lst:
        header = header + f" {keys[1]}_mean"
        for q in quantiles:
            header = header + f" {keys[1]}_q{100*q:.0f}"
    
    processed_data = []

    for n_train in n_train_lst:
        res = concatenate_all_seeds(path + str(n_train) + "_ntest_1000/") 
        normalization = np.mean(res.get('u_test')**2)
        data_line = [n_train]
        for keys in key_lst:
            vals = res.get(keys[0]) / normalization
            vals = np.sqrt(vals)
            data_line.append(np.mean(vals))
            for q in quantiles:
                data_line.append(np.quantile(vals, q))
        processed_data.append(data_line)
    
    processed_data = np.array(processed_data)

    return processed_data, header


def save_fitted_vs_ntrain(bench_name, plotting=False):
    prefix_lst, _, _, _= get_prefix_label_lst(bench_name)
    
    data_path = root_path + 'outputs/' + bench_name + '/'
    processed_path = root_path + 'processed_outputs/' + bench_name + '/'
    
    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
        print(f"Directory {processed_path} created successfully!")
    
    processed_data_list = []

    for prefix in prefix_lst:

        load_name = data_path + prefix
        processed_data, head = get_fitted_vs_ntrain(load_name)
        processed_data_list.append(processed_data)

        save_name = processed_path + prefix + "growing.dat"
        np.savetxt(save_name, processed_data, header=head, comments='')
        print(f"Saved fitted vs ntrain for {bench_name} at {save_name}")
    
    if plotting:
        plot_fitted_vs_ntrain(processed_data_list, head, bench_name)


def plot_fitted_vs_ntrain(processed_data_list, header, bench_name):

    _, _, labels, _ = get_prefix_label_lst(bench_name)

    plotted_keys = [
        'rl2',
        'rsqrtploss',
    ]
    
    fig, ax = plt.subplots(len(plotted_keys),2, figsize=(10,10))
    fig.suptitle(bench_name)

    colors = [f'C0{i}' for i in range(len(labels))]
    for i, processed_data in enumerate(processed_data_list):

        for j, prefix in enumerate(plotted_keys):
            col_names = np.array(str.split(header))

            for l, suffix in enumerate(['train', 'test']):
                key = prefix + '_' + suffix
                col_nam = key + f'_mean'
                ind = np.arange(len(col_names))[col_names == col_nam]
                key_mean = processed_data[:,ind]
                
                key_quantiles = {}
                for q in quantiles:
                    q_str = f"q{100*q:.0f}"
                    col_nam = key + '_' + q_str
                    ind = np.arange(len(col_names))[col_names == col_nam]
                    key_quantiles[q_str] = processed_data[:,ind].reshape(-1)

                ax[j,l].plot(n_train_lst, key_quantiles['q50'], label=labels[i], color=colors[i])
                ax[j,l].fill_between(n_train_lst, key_quantiles['q0'], key_quantiles['q100'], alpha=0.1, color=colors[i])
                ax[j,l].fill_between(n_train_lst, key_quantiles['q10'], key_quantiles['q90'], alpha=0.5, color=colors[i])

    for a in ax.flatten():
        a.set_yscale('log')
        #a.set_xscale('log')
        a.set_xlabel('n_train')
        a.legend()

    for j, key in enumerate(plotted_keys):
        ax[j, 0].set_title(key + '_train')
        ax[j, 1].set_title(key + '_test')

    plt.show()


def save_fitted_vs_ntrain_all(bench_list, plotting):

    for bench in bench_list:
        save_fitted_vs_ntrain(bench, plotting)


def get_cost_hist(path):
    header = "iteration"
    n_iter = 75
    saved_data = np.arange(n_iter).reshape(-1,1)

    for n_train in n_train_lst:
        for q in quantiles:
            header = header + f" n_train_{n_train}_rsqrtcost_q{100*q:.0f}"

        res = concatenate_all_seeds(path + str(n_train) + "_ntest_1000/") 
        cost_hist = np.zeros((n_iter, len(res['script_params'])))
        # repeat last value when needed
        for i, oph in enumerate(res['optim_history']):
            header = header + f" n_train_{n_train}_seed_{i}"
            if len(oph['cost']) > n_iter:
                cost_hist[:, i] = oph['cost'][:n_iter]
            else:
                cost_hist[:, i] = np.concatenate([oph['cost'], oph['cost'][-1] * np.ones(n_iter - len(oph['cost']))])
        
        normalization = np.mean(res.get('u_test')**2)
        cost_hist = (cost_hist / normalization)**0.5
        cost_quant_hist = np.quantile(cost_hist, quantiles, axis=1)
        saved_data = np.hstack([saved_data, cost_quant_hist.T, cost_hist])

    return saved_data, header


def save_cost_hist(bench_name, plotting=False):
    _, prefix_lst, _, _ = get_prefix_label_lst(bench_name)

    data_path = root_path + 'outputs/' + bench_name + '/'
    processed_path = root_path + 'processed_outputs/' + bench_name + '/'

    if not os.path.exists(processed_path):
        os.makedirs(processed_path)
        print(f"Directory {processed_path} created successfully!")

    processed_data_list = []

    for prefix in prefix_lst:
        load_name = data_path + prefix
        processed_data, head = get_cost_hist(load_name)
        processed_data_list.append(processed_data)

        save_name = processed_path + prefix +"cost_hist.dat"
        np.savetxt(save_name, processed_data, header=head, comments='')
        print(f"Saved fitted vs ntrain for {bench_name} at {save_name}")

    if plotting:
        plot_cost_hist(processed_data_list, head, bench_name)


def plot_cost_hist(processed_data_list, header, bench_name):
    
    iterations = processed_data_list[0][:,0]

    _, _, _, labels = get_prefix_label_lst(bench_name)
    
    fig, ax = plt.subplots(len(n_train_lst), 2, figsize=(10,15))
    fig.suptitle(f"{bench_name} - Optimizing Poincare loss")

    colors = [f'C0{i}' for i in range(len(labels))]
    col_names = np.array(str.split(header))
    for i, processed_data in enumerate(processed_data_list):
        for j, n_train in enumerate(n_train_lst):

            # plot the quantiles
            cost_quantiles = {}
            for q in quantiles:
                q_str = f"q{100*q:.0f}"
                col_nam = f'n_train_{n_train}_rsqrtcost_' + q_str
                ind = np.arange(len(col_names))[col_names == col_nam]
                cost_quantiles[q_str] = processed_data[:,ind].reshape(-1)

            ax[j,0].fill_between(iterations, cost_quantiles['q0'], cost_quantiles['q100'], alpha=0.2, color=colors[i], label=labels[i] + ' q0 q100')
            ax[j,1].fill_between(iterations, cost_quantiles['q10'], cost_quantiles['q90'], alpha=0.2, color=colors[i], label=labels[i] + ' q10 q90')
            ax[j,0].set_ylabel(f'Train RsqrtPloss \nntrain={n_train}')

    for a in ax.flatten():
        a.set_yscale('log')
        #a.set_xscale('log')
        a.set_xlabel('iterations')
        a.legend()

    plt.show()


def save_cost_hist_all(bench_list, plotting=False):
    for bench in bench_list:
        save_cost_hist(bench, plotting)

# %%
bench_list = [
    'quartic_sin_collective_d_9_m_2_nmat_3',
    'quartic_sin_collective_d_9_m_3_nmat_3',
    'quartic_sin_collective_d_9_m_3_nmat_4',
    #'quartic_sin_collective_d_17_m_3_nmat_3'
    ]

# %%

plotting = True
save_fitted_vs_ntrain_all(bench_list, plotting=plotting)

# %%
#save_cost_hist_all(bench_list, plotting=plotting)

# %%
DEBUG = False

if DEBUG:
    results = concatenate_all_seeds('./tensap/numerics/outputs/sin_squared_norm_d_8_m_1/fitmethod_pymanopt_init_as_ntrain_100_ntest_1000/')

    ind = results['fitted_mse_test'].argmax()
    cv_features = results['cv_features'][ind]
    cv_profile = results['cv_profile'][ind]

    x_train = results['x_train'][ind]
    x_test = results['x_test'][ind]

    z_train = results['z_train'][ind]
    z_test = results['z_test'][ind]

    y_train = cv_profile.predict(z_train)
    y_test = cv_profile.predict(z_test)

    u_train = results['u_train'][ind]
    u_test = results['u_test'][ind]

    jac_u_train = results['jac_u_train'][ind]
    jac_u_test = results['jac_u_test'][ind]


    m = z_train.shape[1]
    fig, ax = plt.subplots(1 + m, 1, figsize=(5,5*m))
    ax[0].scatter(y_test, u_test, label='test', marker='x')
    ax[0].scatter(y_train, u_train, label='train', marker='+')
    ax[0].set_ylabel('u(X)')
    ax[0].set_xlabel('fog(X)')
    ax[0].legend()

    for i, axi in enumerate(ax[1:]):

        axi.scatter(z_test[:,i], u_test, label='u_test', marker='x')
        axi.scatter(z_train[:,i], u_train, label='u_train', marker='+')
        axi.set_xlabel(f'g_{i}(X)')
        axi.legend()
        
    plt.legend()
    plt.show()
