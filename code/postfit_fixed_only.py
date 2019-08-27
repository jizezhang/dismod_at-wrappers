import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_residuals(path_to_data_csv: str, bins=None):
    residuals = pd.read_csv(path_to_data_csv)['residual'].values
    if bins:
        plt.hist(residuals,bins=bins)
    else:
        bins = np.round(residuals.shape[0]/4.)
        plt.hist(residuals,bins=bins)
    plt.xlabel('residual')
    plt.title('histogram for residuals')
    return residuals

def plot_change_over_age(type: str, name: str, measurement: str, path_to_variable_csv: str,\
                              path_to_data_csv:str, time_list,\
                              time_idx=None, legend=True, ylim = [],curve_per_plot=5,\
                              plot_data=True):
    data = pd.read_csv(path_to_data_csv)
    var = pd.read_csv(path_to_variable_csv)
    res = []
    if type == 'rate':
        res = var[(var['var_type'] == 'rate') & (var['rate']==name)]
    elif type == 'covariate':
        res = var[var['covariate'] == name]
    elif type == 'residual':
        res = data[data['integrand'] == measurement]

    Y = []
    Z = []
    ntimes = len(time_list)
    if type == 'rate' or type == 'covariate':
        res.sort_values(by=['age'])
        Y = res['age'].values.reshape((ntimes,-1),order='F')
        Z = res['fit_value'].values.reshape((ntimes,-1),order='F')
    if time_idx is None:
        time_idx = range(ntimes)

    k = 0
    for i in time_idx:
        data_sub = data[(data['time_lo'] <= time_list[i]) & \
                        (data['time_up'] >= time_list[i]) & \
                        (data['integrand'] == measurement)]
        if type == 'rate' or type == 'covariate':
            if k%curve_per_plot == 0:
                plt.figure()
                plt.title(name+' plot across age')
                if ylim != []:
                    plt.ylim(ylim)
            plt.plot(Y[i,:],Z[i,:],'-',label = "time "+ str(time_list[i]))
            plt.xlabel('age')
            plt.ylabel(type+' '+name)
            plt.title(name+' plot across age')
            k += 1
            if legend:
                plt.legend()
            if plot_data:
                for j,row in data_sub.iterrows():
                    plt.plot([row['age_lo'],row['age_up']],[row['meas_value'], \
                                  row['meas_value']],'-',color='tab:grey',linewidth=.5)
        elif type == 'residual':
            fit_sub = res[(res['time_lo'] <= time_list[i]) & \
                          (res['time_up'] >= time_list[i])]
            if fit_sub.shape[0] > 0:
                plt.figure()
                plt.hist(fit_sub['residual'].values)
                plt.title('residual, data that include year '+str(time_list[i]))
    return Y,Z

def plot_change_over_time(type: str, name: str,measurement: str, path_to_variable_csv: str, \
                               path_to_data_csv: str, \
                               age_list, age_idx=None, legend=True, \
                               ylim=[], curve_per_plot=5, plot_data=True):
    data = pd.read_csv(path_to_data_csv)
    var = pd.read_csv(path_to_variable_csv)
    res = []
    if type == 'rate':
        res = var[(var['var_type'] == 'rate') & (var['rate']==name)]
    elif type == 'covariate':
        res = var[var['covariate'] == name]
    elif type == 'residual':
        res = data[data['integrand'] == measurement]
    X = []
    Z = []
    nages = len(age_list)
    if type == 'rate' or type == 'covariate':
        res.sort_values(by=['age'])
        X = res['time'].values.reshape((nages,-1))
        Z = res['fit_value'].values.reshape((nages,-1))
    if age_idx is None:
        age_idx = range(nages)
    k = 0
    for i in age_idx:
        data_sub = data[(data['age_lo'] <= age_list[i]) & \
                        (data['age_up'] >= age_list[i]) & \
                        (data['integrand'] == measurement)]
        if type == 'rate' or type == 'covariate':
            if k%curve_per_plot == 0:
                plt.figure()
                plt.title(name+' plot across time')
                if ylim != []:
                    plt.ylim(ylim)
            #plt.legend()
            plt.plot(X[i,:],Z[i,:],'-',label = "age "+ str(age_list[i]))
            k += 1
            plt.xlabel('time')
            plt.ylabel(type + ' '+ name)
            plt.title(name+' plot across time')
            if plot_data:
                for j,row in data_sub.iterrows():
                    plt.plot([row['time_lo'],row['time_up']],[row['meas_value'], \
                             row['meas_value']],'-',color='tab:gray',linewidth=.5)
            if legend:
                plt.legend()
        else:
            fit_sub = res[(res['age_lo'] <= age_list[i]) & \
                          (res['age_up'] >= age_list[i])]
            if fit_sub.shape[0] > 0:
                plt.figure()
                plt.hist(fit_sub['residual'].values)
                plt.title('residual, data that include age ' +str(age_list[i]))

    return X,Z
