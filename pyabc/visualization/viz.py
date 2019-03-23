import pyabc
import pyabc.visualization
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import pandas as pd
import pickle
import pymc3


limits = {
    'p_infectCC': (-10, -2),
    'coupling': (-6, -1),
    'p_infectDIFF': (-9, -2),
    'pII': (-4, -2),
    'degrE2': (-2, 1)
}

true_pars = {
    'p_infectCC': np.log10(2e-7),
    'coupling': np.log10(0.01),
    'p_infectDIFF': np.log10(3.16e-5),
    'pII': np.log10(0.001584893),
    'degrE2': np.log10(0.3)
}


def resample(arr, w, n_resampled):
    w = w/ np.sum(w)
    n_par = len(w)  # = len(arr)
    indices = np.random.choice(range(0, n_par), n_resampled, p=w)
    return np.array([arr[i] for i in indices])


def viz_errorbars(label, history, par_list):
    t_start = 0
    n_par = len(par_list)
    n_pop = history.n_populations
    mean = np.empty((n_par, n_pop))
    ci_90 = np.empty((n_par, n_pop, 2))
    ci_95 = np.empty((n_par, n_pop, 2))
    err_90 = np.empty((n_par, n_pop, 2))
    err_95 = np.empty((n_par, n_pop, 2))
    for ip in range(0, n_par):
        for t in range(0, history.max_t + 1):
            df, w = history.get_distribution(m=0, t=t)
            arr = list(df[par_list[ip]])
            arr_resampled = resample(arr, w, int(1e6))
            mean[ip, t] = np.mean(arr_resampled)
            ci_90[ip, t, :] = pymc3.stats.hpd(arr_resampled, alpha=0.1)
            ci_95[ip, t, :] = pymc3.stats.hpd(arr_resampled, alpha=0.05)
            with open("cis.txt", 'a') as f:
                f.write("" + par_list[ip] + "\t " + str(t) + "\t " + str(ci_90[ip, t, :]) + "\t " + str(ci_95[ip, t, :]) + "\n")
            err_90[ip, t, :] = [mean[ip, t] - ci_90[ip, t, 0], ci_90[ip, t, 1] - mean[ip, t]]
            err_95[ip, t, :] = [mean[ip, t] - ci_95[ip, t, 0], ci_95[ip, t, 1] - mean[ip, t]]
    for ip in range(0, n_par):
        plt.figure()
        plt.errorbar(range(t_start, n_pop),
                     mean[ip, t_start:],
                     yerr = np.transpose(err_90[ip, t_start:, :]),
                     capsize=5, ecolor='coral') 
        plt.errorbar(range(t_start, n_pop),
                     mean[ip, t_start:],
                     yerr = np.transpose(err_95[ip, t_start:, :]),
                     capsize=7, ecolor='maroon')
        plt.axhline(y=true_pars[par_list[ip]], color='C1')
        plt.ylim(limits[par_list[ip]])
        plt.title(par_list[ip])
        plt.xlabel("Generation t")
        plt.ylabel("Mean and 90% (orange) and 95% (red) CIs")
        plt.savefig(label + "_errorbars_" + par_list[ip])
        plt.close()


def viz_quantiles(label, history, par_list):
    n_par = len(par_list)
    n_pop = history.n_populations
    mean = np.empty((n_par, n_pop))
    quantiles = np.array([0.025, 0.05, 0.125, 0.25, 0.375, 0.45, 0.5, 0.55, 0.625, 0.75, 0.875, 0.95, 0.975])
    n_quantiles = len(quantiles)
    cis = np.empty((n_par, n_pop, n_quantiles))
    
    df = pd.DataFrame(columns=['time', 'quantile', *par_list])

    # get mean and quantiles
    for t in range(0, history.max_t + 1):
        pop = history.get_population(t=t)
        lst = pop.get_list()
        weights = np.array([pt.weight for pt in lst])
        parameters = [pt.parameter for pt in lst]
        for ipar, par in enumerate(par_list):
            points = np.array([pt.parameter[par] for pt in lst])
            mean_t_ipar = np.sum(weights * points)
            mean[ipar, t] = mean_t_ipar
            for iquantile, quantile in enumerate(quantiles):
                cis[ipar, t, iquantile] = \
                    pyabc.weighted_statistics.weighted_quantile(points, weights, alpha=quantile)
    # plot
    for ipar, par in enumerate(par_list):
        plt.figure()
        for j in range(0, 6):
            plt.errorbar(
                range(0, n_pop),
                mean[ipar, :],
                yerr=[mean[ipar, :] - cis[ipar, :, j], cis[ipar, :, 12-j] - mean[ipar, :]],
                capsize=12-1.5*j,
                label=str(int(np.round(100*(1.0-quantiles[j]*2)))) + "%")
        plt.axhline(y=true_pars[par], color='C1')
        plt.ylim(limits[par])
        plt.title(par)
        plt.xlabel("Generation t")
        plt.ylabel("Mean and quantiles")
        plt.legend()
        plt.savefig(label + "_quantiles_" + par)
        plt.close
    
    # create dataframe
    for t in range(0, history.max_t + 1):
        for iquantile, quantile in enumerate(quantiles):
            dct = dict(zip(['time', 'quantile', *par_list], [t, quantile, *list(cis[:, t, iquantile])]))
            df = df.append(dct, ignore_index=True)
    print(df)
    df.to_csv("quantiles.tsv", sep='\t')


def viz_kdes(label, history):
    for t in range(0, history.max_t + 1):
        df, w = history.get_distribution(m=0, t=t)
        ax = pyabc.visualization.plot_kde_matrix(
            df, w,
            limits=limits,
            colorbar=False,
            refval=true_pars)
        plt.savefig(label + "_kde_matrix_" + str(t))
        plt.close()


def viz_eps(label, history):
    eps = np.array(history.get_all_populations()['epsilon'])
    plt.figure()
    plt.plot(np.log(eps[1:]), 'x-')
    plt.xlabel("Generation t")
    plt.ylabel("Log(Epsilon)")
    plt.savefig(label + "_eps")


def viz_samples(label, history):
    samples = np.array(history.get_all_populations()['samples'])
    plt.figure()
    plt.plot(np.log(samples[1:]), 'x-')
    plt.xlabel("Generation t")
    plt.ylabel("Log(#Samples)")
    plt.savefig(label + "_samples")


db_file = "sqlite:///results_20190304/hcv_sili_20190304.db"
#db_file = "sqlite:///results_exp29d/Exp29D.db"
history = pyabc.History(db_file)

viz_kdes("viz", history)
viz_errorbars("viz", history, list(true_pars.keys()))
viz_quantiles("viz", history, list(true_pars.keys()))
viz_eps("viz", history)
viz_samples("viz", history)
