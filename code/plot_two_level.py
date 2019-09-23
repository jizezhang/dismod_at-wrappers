import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import sqlite3


class PlotTwoLevel:

    def __init__(self, path_to_db, path_to_data_csv, age_list, time_list):
        self.path_to_db = path_to_db
        self.path_to_data = path_to_data_csv
        self.age_list = age_list
        self.time_list = time_list

        self.integrand_values = self.get_integrand_values()
        self.data_values = pd.read_csv(self.path_to_data)

        self.rate_to_integrand = {'iota': 'Sincidence', 'rho': 'remission', 'chi': 'mtexcess', 'omega': 'mtother'}

    def get_integrand_values(self):
        conn = sqlite3.connect(self.path_to_db)
        df = pd.read_sql_query("select node.node_name, integrand.integrand_name, age_lower, age_upper, time_lower, \
                                time_upper, avg_integrand \
                                from avgint \
                                inner join predict on avgint.avgint_id == predict.avgint_id \
                                inner join node on node.node_id == avgint.node_id \
                                inner join integrand on avgint.integrand_id == integrand.integrand_id;", conn)
        return df

    def plot_residuals(self, location: str, bins: int = None):
        residuals = self.data_values['residual'].values
        if location != 'all':
            residuals = data[data['node'] == location]['residual'].values
        if bins:
            plt.hist(residuals, bins=bins)
        else:
            bins = int(np.round(residuals.shape[0]/4.))
            plt.hist(residuals, bins=bins)
        plt.xlabel('residual')
        plt.title('histogram for residuals, ' + location)

    def plot_change_over_age(self, type: str, name: str, measurement: str, location: str,
                             time_idx: List[int] = None, legend: bool = True, ylim: List[float] = None,
                             curve_per_plot: int = 5, plot_data: bool = True):

        data = self.data_values
        if location != 'all':
            data = self.data_values[self.data_values['node'] == location]

        var = []
        if type == 'rate':
            var = self.integrand_values[self.integrand_values['integrand_name'] == self.rate_to_integrand[rate]]
        elif type == 'covariate':
            res = self.integrand_values[self.integrand_values['integrand_name'] == name]

        ntimes = len(self.time_list)
        var.sort_values(by=['age_lower'])
        Y = var['age_lower'].values.reshape((ntimes, -1), order='F')
        Z = var['avg_integrand'].values.reshape((ntimes, -1), order='F')
        if time_idx is None:
            time_idx = range(ntimes)

        k = 0
        for i in time_idx:
            data_sub = data[(data['time_lo'] <= time_list[i]) & \
                            (data['time_up'] >= time_list[i]) & \
                            (data['integrand'] == measurement)]
            if k % curve_per_plot == 0:
                plt.figure()
                plt.title(name+' plot across age')
                if ylim is not None:
                    plt.ylim(ylim)
            plt.plot(Y[i, :], Z[i, :], '-', label="time " + str(time_list[i]))
            plt.xlabel('age')
            plt.ylabel(type+' '+name)
            plt.title(name+' plot across age')
            k += 1
            if legend:
                plt.legend()
            if plot_data:
                for j, row in data_sub.iterrows():
                    plt.plot([row['age_lo'], row['age_up']],
                             [row['meas_value'], row['meas_value']], '-', color='tab:grey', linewidth=.5)
                    if row['age_lo'] == row['age_up']:
                        plt.plot(row['age_lo'], row['meas_value'], '.', color='tab:gray', markersize=5)

# def plot_change_over_age(type: str, name: str, measurement: str, location: str,
#                          path_to_variable_csv: str,
#                          path_to_data_csv: str, time_list: List,
#                          time_idx: List[int] = None, legend: bool = True, ylim: List[float] = None,
#                          curve_per_plot: int = 5, plot_data: bool = True):
#     data = pd.read_csv(path_to_data_csv)
#     if location != 'world':
#         data = data[data['node'] == location]
#     var_all = pd.read_csv(path_to_variable_csv)
#     if type == 'rate':
#         var = var_all[var_all['node'] == location]
#     else:
#         var = var_all
#     res = []
#     if type == 'rate':
#         res = var[(var['var_type'] == 'rate') & (var['rate']==name)]
#     elif type == 'covariate':
#         res = var[var['covariate'] == name]
#     elif type == 'residual':
#         res = data[data['integrand'] == measurement]
#
#     Y = []
#     Z = []
#     ntimes = len(time_list)
#     if type == 'rate' or type == 'covariate':
#         res.sort_values(by=['age'])
#         Y = res['age'].values.reshape((ntimes, -1),order='F')
#         Z = res['fit_value'].values.reshape((ntimes, -1), order='F')
#         if type == 'rate' and location != 'world':
#             var_world = var_all[var_all['node'] == 'world']
#             base_rate = var_world['fit_value'].values.reshape((ntimes, -1), order='F')
#             Z = np.exp(Z)*base_rate
#     if time_idx is None:
#         time_idx = range(ntimes)
#
#     k = 0
#     for i in time_idx:
#         data_sub = data[(data['time_lo'] <= time_list[i]) & \
#                         (data['time_up'] >= time_list[i]) & \
#                         (data['integrand'] == measurement)]
#         if type == 'rate' or type == 'covariate':
#             if k%curve_per_plot == 0:
#                 plt.figure()
#                 plt.title(name+' plot across age')
#                 if ylim is not None:
#                     plt.ylim(ylim)
#             plt.plot(Y[i, :], Z[i, :], '-', label="time " + str(time_list[i]))
#             plt.xlabel('age')
#             plt.ylabel(type+' '+name)
#             plt.title(name+' plot across age')
#             k += 1
#             if legend:
#                 plt.legend()
#             if plot_data:
#                 for j, row in data_sub.iterrows():
#                     plt.plot([row['age_lo'], row['age_up']],
#                              [row['meas_value'], row['meas_value']], '-', color='tab:grey', linewidth=.5)
#                     if row['age_lo'] == row['age_up']:
#                         plt.plot(row['age_lo'], row['meas_value'], '.', color='tab:gray', markersize=5)
#         elif type == 'residual':
#             fit_sub = res[(res['time_lo'] <= time_list[i]) &
#                           (res['time_up'] >= time_list[i])]
#             if fit_sub.shape[0] > 0:
#                 plt.figure()
#                 plt.hist(fit_sub['residual'].values)
#                 plt.title('residual, data that include year '+str(time_list[i]))
#     return Y, Z


def plot_change_over_time(type: str, name: str, measurement: str, location: str,
                          path_to_variable_csv: str,
                          path_to_data_csv: str,
                          age_list, age_idx: List[int] = None, legend=True,
                          ylim: List[float] = None, curve_per_plot: int = 5, plot_data: bool = True):
    data = pd.read_csv(path_to_data_csv)
    if location != 'world':
        data = data[data['node'] == location]
    var_all = pd.read_csv(path_to_variable_csv)
    if type == 'rate':
        var = var_all[var_all['node'] == location]
    else:
        var = var_all
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
        if type == 'rate' and location != 'world':
            var_world = var_all[var_all['node'] == 'world']
            base_rate = var_world['fit_value'].values.reshape((nages, -1))
            Z = np.exp(Z)*base_rate
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
                if ylim is not None:
                    plt.ylim(ylim)
            plt.plot(X[i,:],Z[i,:],'-',label = "age "+ str(age_list[i]))
            k += 1
            plt.xlabel('time')
            plt.ylabel(type + ' '+ name)
            plt.title(name+' plot across time')
            if plot_data:
                for j, row in data_sub.iterrows():
                    plt.plot([row['time_lo'], row['time_up']],
                             [row['meas_value'], row['meas_value']], '-', color='tab:gray', linewidth=.5)
                    if row['time_lo'] == row['time_up']:
                        plt.plot(row['time_lo'], row['meas_value'], '.', color='tab:gray', markersize=5)
            if legend:
                plt.legend()
        else:
            fit_sub = res[(res['age_lo'] <= age_list[i]) &
                          (res['age_up'] >= age_list[i])]
            if fit_sub.shape[0] > 0:
                plt.figure()
                plt.hist(fit_sub['residual'].values)
                plt.title('residual, data that include age ' +str(age_list[i]))

    return X, Z
