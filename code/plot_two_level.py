import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from dismod_output import DismodOutput
import dismod_at
import os

program = '/home/prefix/dismod_at.release/bin/dismod_at'

def sigmoid(x):
    return 1./(1. + np.exp(-x))


class PlotTwoLevel:

    def __init__(self, path_to_folder, db_file_name):
        self.path_to_db = os.path.join(path_to_folder + db_file_name)
        dismod_at.db2csv_command(self.path_to_db)
        self.db_output = DismodOutput(self.path_to_db)

        self.path_to_data = path_to_folder + 'data.csv'
        self.age_list, self.time_list = self.db_output.get_age_time_lists()

        self.integrand_values = self.db_output.get_integrand_values()
        self.data_values = pd.read_csv(self.path_to_data)
        self.cov_name_to_id = self.db_output.get_covarates_names()

        self.rate_to_integrand = {'iota': 'Sincidence', 'rho': 'remission', 'chi': 'mtexcess', 'omega': 'mtother'}

    # def get_age_time_lists(self):
    #     conn = sqlite3.connect(self.path_to_db)
    #     age = pd.read_sql_query("select age from age;", conn)
    #     time = pd.read_sql_query("select time from time", conn)
    #     conn.close()
    #     return age.values.squeeze(), time.values.squeeze()
    #
    # def get_covarates(self):
    #     conn = sqlite3.connect(self.path_to_db)
    #     df = pd.read_sql_query("select covariate.covariate_id, covariate_name, mulcov_type \
    #                             from covariate \
    #                             inner join mulcov on covariate.covariate_id == mulcov.covariate_id;", conn)
    #     cov_name_to_id = {}
    #     for i, row in df.iterrows():
    #         if row['mulcov_type'] == 'rate_value':
    #             cov_name_to_id[row['covariate_name']] = 'mulcov_' + str(row['covariate_id'])
    #     conn.close()
    #
    #     return cov_name_to_id
    #
    # def get_integrand_values(self):
    #     conn = sqlite3.connect(self.path_to_db)
    #     df = pd.read_sql_query("select node.node_name, integrand.integrand_name, age_lower, age_upper, time_lower, \
    #                             time_upper, avg_integrand \
    #                             from avgint \
    #                             inner join predict on avgint.avgint_id == predict.avgint_id \
    #                             inner join node on node.node_id == avgint.node_id \
    #                             inner join integrand on avgint.integrand_id == integrand.integrand_id;", conn)
    #     conn.close()
    #     return df

    def plot_residuals(self, location: str, measurement: str, bins: int = None):
        residuals = self.data_values[self.data_values['integrand'] == measurement]['residual'].values
        if location != 'all':
            residuals = self.data_values[self.data_values['node'] == location]['residual'].values
        if bins:
            plt.hist(residuals, bins=bins)
        else:
            bins = int(np.round(residuals.shape[0]/4.))
            plt.hist(residuals, bins=bins)
        plt.xlabel('residual')
        plt.title('histogram for residuals, ' + location)

    def plot_change_over_age(self, type: str, name: str, measurement: str, locations: str,
                             time_idx: List[int] = None, legend: bool = True, ylim: List[float] = None,
                             curve_per_plot: int = 5, plot_data: bool = True):
        values = []
        for location in locations:
            var = []
            if type == 'rate':
                var = self.integrand_values[(self.integrand_values['integrand_name'] == self.rate_to_integrand[name]) &
                                            (self.integrand_values['node_name'] == location)]
            #elif type == 'covariate':
            #    var = self.integrand_values[(self.integrand_values['integrand_name'] == self.cov_name_to_id[name]) &
            #                                (self.integrand_values['node_name'] == location)]

            ntimes = len(self.time_list)
            var.sort_values(by=['age_lower'])
            #Y = var['age_lower'].values.reshape((ntimes, -1), order='F')
            Z = var['avg_integrand'].values.squeeze().reshape((ntimes, -1), order='F')
            values.append(Z)

        if time_idx is None:
            time_idx = range(ntimes)

        k = 0
        for i in time_idx:
            if k % curve_per_plot == 0:
                fig, axes = plt.subplots(1, len(locations), sharey=True, figsize=(len(locations)*5, 3))
            k += 1
            for loc_i in range(len(locations)):
                ax = None
                if len(locations) == 1:
                    ax = axes
                else:
                    ax = axes[loc_i]
                location = locations[loc_i]
                if ylim is not None:
                    ax.set_ylim(ylim)
                data = self.data_values
                if location != 'all':
                    data = self.data_values[self.data_values['node'] == location]
                data_sub = data[(data['time_lo'] <= self.time_list[i]) &
                                (data['time_up'] >= self.time_list[i]) &
                                (data['integrand'] == measurement)]
                Z = values[loc_i]
                #print(self.age_list, Z.shape)
                ax.plot(self.age_list, Z[i, :], '-', label="time " + str(self.time_list[i]))
                ax.set_xlabel('age')
                ax.set_ylabel(type+' '+name)
                ax.set_title(location + ": " + name+' plot across age')
                if legend:
                    ax.legend()
                if plot_data:
                    for j, row in data_sub.iterrows():
                        color = 'tab:grey'
                        if np.abs(row['residual']) >= 3.:
                            color = 'rosybrown'
                        ax.plot([row['age_lo'], row['age_up']],
                                [row['meas_value'], row['meas_value']], '-', color=color,
                                linewidth=sigmoid(-row['meas_std']))
                        if row['age_lo'] == row['age_up']:
                            ax.plot(row['age_lo'], row['meas_value'], '.', color=color,
                                    markersize=5*sigmoid(-row['meas_std']))


    def plot_change_over_time(self, type: str, name: str, measurement: str, locations: str,
                             age_idx: List[int] = None, legend: bool = True, ylim: List[float] = None,
                             curve_per_plot: int = 5, plot_data: bool = True):

        values = []
        for location in locations:
            var = []
            if type == 'rate':
                var = self.integrand_values[(self.integrand_values['integrand_name'] == self.rate_to_integrand[name]) &
                                            (self.integrand_values['node_name'] == location)]
            #elif type == 'covariate':
            #    var = self.integrand_values[(self.integrand_values['integrand_name'] == self.cov_name_to_id[name]) &
            #                                (self.integrand_values['node_name'] == location)]

            nages = len(self.age_list)
            var.sort_values(by=['age_lower'])
            Z = var['avg_integrand'].values.squeeze().reshape((nages, -1))
            values.append(Z)

        if age_idx is None:
            age_idx = range(nages)

        k = 0
        for i in age_idx:
            if k % curve_per_plot == 0:
                fig, axes = plt.subplots(1, len(locations), sharey=True, figsize=(len(locations) * 5, 3))
            k += 1
            for loc_i in range(len(locations)):
                if len(locations) == 1:
                    ax = axes
                else:
                    ax = axes[loc_i]
                location = locations[loc_i]
                if ylim is not None:
                    ax.set_ylim(ylim)
                data = self.data_values
                if location != 'all':
                    data = self.data_values[self.data_values['node'] == location]
                data_sub = data[(data['age_lo'] <= self.age_list[i]) &
                                (data['age_up'] >= self.age_list[i]) &
                                (data['integrand'] == measurement)]
                Z = values[loc_i]
                ax.plot(self.time_list, Z[i, :], '-', label="age " + str(self.age_list[i]))
                ax.set_xlabel('year')
                ax.set_ylabel(type+' '+name)
                ax.set_title(location + ": " + name+' plot across time')
                k += 1
                if legend:
                    ax.legend()
                if plot_data:
                    for j, row in data_sub.iterrows():
                        color = 'tab:grey'
                        if row['residual'] >= 3.:
                            color = 'rosybrown'
                        ax.plot([row['time_lo'], row['time_up']],
                                [row['meas_value'], row['meas_value']], '-', color=color,
                                linewidth=sigmoid(-row['meas_std']))
                        if row['time_lo'] == row['time_up']:
                            ax.plot(row['time_lo'], row['meas_value'], '.', color=color,
                                    markersize=5*sigmoid(-row['meas_std']))
