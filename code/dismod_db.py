import numpy as np
import sys
import subprocess
import copy
import pandas as pd
from typing import Dict, List, Any, Tuple
import dismod_at

program = '/home/prefix/dismod_at.release/bin/dismod_at'


class DismodDB:

    def __init__(self, data: pd.DataFrame, location_names: List[str],
                 integrand: List[str], rates: List[str],
                 rate_parent_priors: List[Tuple[Dict[str, str], ...]],
                 rate_child_priors: List[Tuple[Dict[str, str], ...]],
                 meas_noise_density: Dict[str, Dict[str, Any]],
                 path_to_db: str,
                 covariates: List[Dict[str, str]] = None,
                 cov_priors: List[Tuple[Dict[str, str], ...]] = None,
                 age_list: List[int] = None, time_list: List[int] = None,
                 sparse_child_grid: bool = True):

        """
        """

        self.density_dict = {'uniform': 0, 'gaussian': 1, 'laplace': 2,
                             'students': 3, 'log_gaussian': 4,
                             'log_laplace': 5, 'log_students': 6}
        self.location_names = location_names
        self.integrand = integrand
        self.meas_noise_density = meas_noise_density
        self.rates = rates
        self.data = data
        self.n = self.data.shape[0]
        if covariates is not None:
            self.m = len(covariates)
            self.covariates = covariates
        else:
            self.m = 0
            self.covariates = []
        self.rate_parent_priors = rate_parent_priors
        self.rate_child_priors = rate_child_priors
        if cov_priors is not None:
            self.cov_priors = cov_priors
        else:
            self.cov_priors = []

        self.path = path_to_db

        if age_list is not None:
            self.age_list = age_list
        else:
            self.age_list = self.create_age_list()
        if time_list is not None:
            self.time_list = time_list
        else:
            self.time_list = self.create_time_list()

        self.sparse_child_grid = sparse_child_grid

        self.check()
        self.create_tables()

    def check(self):
        assert all([x in self.meas_noise_density for x in self.integrand])
        assert self.m == len(self.cov_priors)
        assert len(self.rates) == len(self.rate_parent_priors)
        assert (len(self.rates) == len(self.rate_child_priors) or len(self.rate_child_priors) == 1)

    def create_tables(self):
        self.create_cov_table()
        self.create_mulcov_table()
        self.create_data_table()
        self.create_smooth_table(self.sparse_child_grid)
        self.create_prior_table()
        self.create_option_table()
        self.create_avgint_table()  # will add mulcov_id to self.integrand
        self.create_default_tables()

    def create_age_list(self):
        max_age = -float('inf')
        min_age = float('inf')
        for i in range(self.n):
            if self.data.loc[i, 'measure'] in self.integrand and \
                    self.data.loc[i, 'location_name'] in self.location_names:
                max_age = max(max_age, self.data.loc[i, 'age_end'])
                min_age = min(min_age, self.data.loc[i, 'age_start'])
        age_list = [int(round(x)) for x in np.linspace(min_age, max_age,
                                                       round((max_age - min_age) / 5) + 1)]
        age_list = sorted(list(set(age_list)))
        return age_list

    def create_time_list(self):
        max_time = -float('inf')
        min_time = float('inf')
        for i in range(self.n):
            if self.data.loc[i, 'measure'] in self.integrand and \
                    self.data.loc[i, 'location_name'] in self.location_names:
                max_time = max(max_time, self.data.loc[i, 'year_end'])
                min_time = min(min_time, self.data.loc[i, 'year_start'])
        time_list = [int(round(x)) for x in np.linspace(min_time, max_time,
                                                        round((max_time - min_time) / 3 + 1))]
        time_list = sorted(list(set(time_list)))
        return time_list

    def create_cov_table(self):
        self.covariate_table = [{'name': 'gamma_one', 'reference': 0.0}, {'name': 'intercept', 'reference': 0.0}]
        for cov in self.covariates:
            self.covariate_table.append({'name': cov['name'], 'reference': 0.0})

    def create_mulcov_table(self):
        self.mulcov_table = [{'covariate': 'gamma_one', 'type': 'meas_noise',
                         'effected': 'Sincidence', 'smooth': 'smooth_gamma_one'},
                        {'covariate': 'intercept', 'type': 'rate_value',
                         'effected': 'iota', 'smooth': 'smooth_intercept'}]
        for cov in self.covariates:
            self.mulcov_table.append({'covariate': cov['name'], 'type': cov['type'],
                                 'effected': cov['effected'], 'smooth': 'smooth_mulcov_' + cov['name']})

    def create_data_table(self):
        self.data_table = list()
        row = {
            'weight': 'constant',
            'gamma_one': 1.0,
            'intercept': 1.0,
            'hold_out': False,
        }
        for data_id in range(self.n):
            if self.data.loc[data_id, 'measure'] in self.integrand and \
                    self.data.loc[data_id, 'location_name'] in self.location_names:
                row['node'] = self.data.loc[data_id, 'location_name']
                row['integrand'] = self.data.loc[data_id, 'measure']
                row.update(self.meas_noise_density[row['integrand']])
                if 'hold_out' in self.data.columns:
                    row['hold_out'] = self.data.loc[data_id, 'hold_out']
                row['meas_value'] = self.data.loc[data_id, 'mean']
                row['meas_std'] = self.data.loc[data_id, 'standard_error']
                row['age_lower'] = self.data.loc[data_id, 'age_start']
                row['age_upper'] = self.data.loc[data_id, 'age_end']
                row['time_lower'] = self.data.loc[data_id, 'year_start']
                row['time_upper'] = self.data.loc[data_id, 'year_end']
                for cov in self.covariates:
                    row[cov['name']] = self.data.loc[data_id, cov['name']]
                self.data_table.append(copy.copy(row))

    def create_avgint_table(self):

        rate_to_integrand = {'iota': 'Sincidence', 'rho': 'remission', 'chi': 'mtexcess', 'omega': 'mtother'}

        used = set()

        self.avgint_table = []
        row = {'intercept': 1.0, 'weight': 'constant', 'gamma_one': 1.0}
        row.update({cov['name']: 0.0 for cov in self.covariates})
        for rate in self.rates:
            for age in self.age_list:
                for time in self.time_list:
                    for loc in self.location_names:
                        row['integrand'] = rate_to_integrand[rate]
                        used.add(rate_to_integrand[rate])
                        row['node'] = loc
                        row['age_lower'] = age
                        row['age_upper'] = age
                        row['time_lower'] = time
                        row['time_upper'] = time
                        self.avgint_table.append(copy.copy(row))
                    row['node'] = 'all'
                    self.avgint_table.append(copy.copy(row))

        for i in range(len(self.covariates)):
            assert self.mulcov_table[i+2]['covariate'] == self.covariates[i]['name']
            self.integrand.append('mulcov_' + str(i+2))

        for integrand in self.integrand:
            if integrand not in used:
                for age in self.age_list:
                    for time in self.time_list:
                        for loc in self.location_names:
                            row['integrand'] = integrand
                            row['node'] = loc
                            row['age_lower'] = age
                            row['age_upper'] = age
                            row['time_lower'] = time
                            row['time_upper'] = time
                            self.avgint_table.append(copy.copy(row))
                        row['node'] = 'all'
                        self.avgint_table.append(copy.copy(row))

        # for i in range(len(self.covariates)):
        #     for age in self.age_list:
        #         for time in self.time_list:
        #             for loc in self.location_names:
        #                 assert self.mulcov_table[i+2]['covariate'] == self.covariates[i]['name']
        #                 row['integrand'] = 'mulcov_' + str(i+2)
        #                 row['node'] = loc
        #                 row['age_lower'] = age
        #                 row['age_upper'] = age
        #                 row['time_lower'] = time
        #                 row['time_upper'] = time
        #                 self.avgint_table.append(copy.copy(row))
        #             row['node'] = 'all'
        #             self.avgint_table.append(copy.copy(row))
        #             #print(self.avgint_table[-1])

    def create_smooth_table(self, sparse_child_grid: bool = True):
        self.smooth_table = [{'name': 'smooth_gamma_one',
                              'age_id': [int(len(self.age_list)/2)],
                              'time_id': [int(len(self.time_list)/2)],
                              'fun': lambda a, t: ('prior_gamma_one', None, None)},
                             {'name': 'smooth_intercept',
                              'age_id': [int(len(self.age_list)/2)],
                              'time_id': [int(len(self.time_list)/2)],
                              'fun': lambda a, t: ('prior_intercept', None, None)}]
        for rate in self.rates:
            self.smooth_table.append({'name': 'smooth_rate_' + rate,
                                      'age_id': range(len(self.age_list)), 'time_id': range(len(self.time_list)),
                                      'fun': lambda a, t, r=rate: ('value_prior_' + r,
                                                                   'dage_prior_' + r, 'dtime_prior_' + r)})
            # use sparse grid for child random effects
            if sparse_child_grid:
                self.smooth_table.append({'name': 'smooth_rate_child_' + rate,
                                          'age_id': [0, len(self.age_list) - 1], 'time_id': [0, len(self.time_list) - 1],
                                          'fun': lambda a, t, r=rate: ('value_prior_child_' + r,
                                                                       'dage_prior_child_' + r, 'dtime_prior_child_' + r)})
            else:
                self.smooth_table.append({'name': 'smooth_rate_child_' + rate,
                                          'age_id': range(len(self.age_list)),
                                          'time_id': range(len(self.time_list)),
                                          'fun': lambda a, t, r=rate: ('value_prior_child_' + r,
                                                                       'dage_prior_child_' + r,
                                                                       'dtime_prior_child_' + r)})

        for cov in self.covariates:
            name = cov['name']
            self.smooth_table.append({'name': 'smooth_mulcov_' + cov['name'],
                                      'age_id': range(len(self.age_list)), 'time_id': range(len(self.time_list)),
                                      'fun': lambda a, t, name=name: ('value_prior_' + name, 'dage_prior_' + name,
                                                                      'dtime_prior_' + name)})

    def create_prior_table(self):
        self.prior_table = [{'name': 'prior_gamma_one', 'density': 'uniform',
                             'lower': 0.0, 'mean': 0.0, 'upper': 0.0},
                            {'name': 'prior_intercept', 'density': 'uniform',
                             'lower': 0.0, 'mean': 0.0, 'upper': 0.0}]
        for i in range(len(self.rates)):
            self.prior_table.append({'name': 'value_prior_' + self.rates[i]})
            self.prior_table[-1].update(self.rate_parent_priors[i][0])
            self.prior_table.append({'name': 'dage_prior_' + self.rates[i]})
            self.prior_table[-1].update(self.rate_parent_priors[i][1])
            self.prior_table.append({'name': 'dtime_prior_' + self.rates[i]})
            self.prior_table[-1].update(self.rate_parent_priors[i][2])

            self.prior_table.append({'name': 'value_prior_child_' + self.rates[i]})
            self.prior_table[-1].update(self.rate_child_priors[i][0])
            self.prior_table.append({'name': 'dage_prior_child_' + self.rates[i]})
            self.prior_table[-1].update(self.rate_child_priors[i][1])
            self.prior_table.append({'name': 'dtime_prior_child_' + self.rates[i]})
            self.prior_table[-1].update(self.rate_child_priors[i][2])

        for i in range(len(self.covariates)):
            self.prior_table.append({'name': 'value_prior_' + self.covariates[i]['name']})
            self.prior_table[-1].update(self.cov_priors[i][0])
            self.prior_table.append({'name': 'dage_prior_' + self.covariates[i]['name']})
            self.prior_table[-1].update(self.cov_priors[i][1])
            self.prior_table.append({'name': 'dtime_prior_' + self.covariates[i]['name']})
            self.prior_table[-1].update(self.cov_priors[i][2])

    def create_option_table(self):
        self.option_table = [
            {'name': 'parent_node_name', 'value': 'all'},
            {'name': 'ode_step_size', 'value': '10.0'},
            {'name': 'quasi_fixed', 'value': 'false'},
            {'name': 'max_num_iter_fixed', 'value': '200'},
            {'name': 'print_level_fixed', 'value': '5'},
            {'name': 'tolerance_fixed', 'value': '1e-4'},
            {'name': 'meas_noise_effect', 'value': 'add_var_scale_all'},
        ]

        if self.integrand == ['Sincidence']:
            self.option_table.append({'name': 'rate_case', 'value': 'iota_pos_rho_zero'})
        elif self.integrand == ['remission']:
            self.option_table.append({'name': 'rate_case', 'value': 'iota_zero_rho_pos'})
        else:
            self.option_table.append({'name': 'rate_case', 'value': 'iota_pos_rho_pos'})

        self.option_name_id = {}
        for i in range(len(self.option_table)):
            self.option_name_id[self.option_table[i]['name']] = i

    def create_default_tables(self):

        self.integrand_table = []
        for intg in self.integrand:
            self.integrand_table.append({'name': intg})

        self.node_table = [{'name': 'all', 'parent': ''}]
        for loc in self.location_names:
            self.node_table.append({'name': loc, 'parent': 'all'})

        self.weight_table = [{'name': 'constant', 'age_id': range(len(self.age_list)),
                         'time_id': range(len(self.time_list)), 'fun': lambda a, t: 1.0}]

        self.rate_table = list()
        for rate in self.rates:
            self.rate_table.append({'name': rate,
                               'parent_smooth': 'smooth_rate_' + rate,
                               'child_smooth': 'smooth_rate_child_' + rate})

    def set_tol(self, tol: float):
        self.option_table[self.option_name_id['tolerance_fixed']]['value'] = str(tol)

    def set_max_iteration(self, max_iter: int):
        self.option_table[self.option_name_id['max_num_iter_fixed']]['value'] = str(max_iter)

    def set_print_level(self, print_level: int):
        self.option_table[self.option_name_id['print_level_fixed']]['value'] = str(print_level)

    def set_zero_sum_constraint(self):
        if 'zero_sum_random' not in self.option_name_id:
            self.option_table.append({'name': 'zero_sum_random', 'value': ' '.join(self.rates)})
            n = len(self.option_name_id)
            self.option_name_id['zero_sum_random'] = n

    def set_meas_density(self, density: str):
        for i in range(len(self.data_table)):
            self.data_table[i]['density'] = density

    def reset_meas_density(self):
        for i, row in enumerate(self.data_table):
            self.data_table[i].update(self.meas_noise_density[row['integrand']])

    def init_database(self, db2csv: bool = True):

        dismod_at.create_database(
            self.path,
            self.age_list,
            self.time_list,
            self.integrand_table,
            self.node_table,
            self.weight_table,
            self.covariate_table,
            self.avgint_table,
            self.data_table,
            self.prior_table,
            self.smooth_table,
            list(),
            self.rate_table,
            self.mulcov_table,
            self.option_table
        )

        command = [program, self.path, 'init']
        print(' '.join(command))
        flag = subprocess.call(command)
        if flag != 0:
            sys.exit('The dismod_at init command failed')
        if db2csv is True:
            dismod_at.db2csv_command(self.path)
