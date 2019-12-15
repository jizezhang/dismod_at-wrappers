import numpy as np
import sys
import subprocess
import copy
import pandas as pd
from typing import Dict, List, Any, Tuple
import dismod_at
import os

program = '/home/prefix/dismod_at.release/bin/dismod_at'

AGE_LIST = list(range(0, 10, 2)) + list(range(10, 130, 5))


class DismodDB:

    def __init__(self, data: pd.DataFrame, location_names: List[str],
                 integrand: List[str], rates: List[str],
                 rate_parent_priors: List[Tuple[Dict[str, str], ...]],
                 rate_child_priors: List[Tuple[Dict[str, str], ...]],
                 meas_noise_density: Dict[str, Dict[str, Any]],
                 path_to_db: str,
                 group_value_prior: Dict[str, str] = None,
                 covariates: List[Tuple[Dict[str, str], List[int]]] = None,
                 cov_priors: List[Dict[str, str]] = None,
                 age_list: List[int] = None, time_list: List[int] = None,
                 child_grid: str = "one"):

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
        if 'group' not in self.data.columns or len(set(self.data['group'].values)) == 1:
            self.data['group'] = 'all'
            self.groups = ['all']
        else:
            self.groups = set(self.data['group'].values)

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

        self.subgroup_value_prior = {'name': 'prior_intercept_sub', 'density': 'uniform',
                                     'lower': 0.0, 'upper': 0.0, 'mean': 0.0}
        if group_value_prior is not None:
            self.subgroup_value_prior = {'name': 'prior_intercept_sub'}
            self.subgroup_value_prior.update(group_value_prior)

        self.path = path_to_db

        if age_list is not None:
            self.age_list = age_list
        else:
            self.age_list = self.create_age_list()
        if time_list is not None:
            self.time_list = time_list
        else:
            self.time_list = self.create_time_list()

        self.child_grid = child_grid

        self.check()
        self.create_tables()

    def check(self):
        #assert all([x in self.meas_noise_density for x in self.integrand])
        assert self.m == len(self.cov_priors)
        assert len(self.rates) == len(self.rate_parent_priors)
        assert (len(self.rates) == len(self.rate_child_priors) or len(self.rate_child_priors) == 1)

    def create_tables(self):
        self.create_cov_table()
        self.create_mulcov_table()
        self.create_data_table()
        self.create_smooth_table(self.child_grid)
        self.create_prior_table()
        self.create_option_table()
        self.create_avgint_table()  # will add mulcov_id to self.integrand
        #self.avgint_table = []
        self.create_default_tables()

    def create_age_list(self):
        max_age = -float('inf')
        min_age = float('inf')
        for i in range(self.n):
            if self.data.loc[i, 'measure'] in self.integrand and \
                    self.data.loc[i, 'location_name'] in self.location_names:
                max_age = max(max_age, self.data.loc[i, 'age_end'])
                min_age = min(min_age, self.data.loc[i, 'age_start'])

        age_list = []
        for i in range(len(AGE_LIST)):
            if min_age <= AGE_LIST[i] <= max_age or (AGE_LIST[i] < min_age <= AGE_LIST[i + 1]):
                age_list.append(AGE_LIST[i])
            elif AGE_LIST[i - 1] <= max_age < AGE_LIST[i]:
                age_list.append(AGE_LIST[i])
                break
        #age_list = [int(round(x)) for x in np.linspace(min_age, max_age,
        #                                               round((max_age - min_age) / 5) + 1)]
        #age_list = sorted(list(set(age_list)))
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

    # def create_subgroup_table(self):
    #     self.subgroup_table = [{'subgroup': 'all', 'group': 'all'}]
    #     if len(self.groups) > 1:
    #         self.subgroup_table.append([{'subgroup': group, 'group': 'all'} for group in self.groups])

    def create_cov_table(self):
        self.covariate_table = [{'name': 'gamma_one', 'reference': 0.0}, {'name': 'intercept', 'reference': 0.0}]
        for cov in self.covariates:
            self.covariate_table.append({'name': cov[0]['name'], 'reference': 0.0})

    def create_mulcov_table(self):
        self.mulcov_table = [{'covariate': 'gamma_one', 'type': 'meas_noise',
                              'effected': 'Sincidence', 'group': 'all',
                              'smooth': 'smooth_gamma_one'}]
        for integrand in self.integrand:
            self.mulcov_table.append({'covariate': 'intercept', 'type': 'meas_value',
                              'effected': integrand, 'group': 'all',
                              'smooth': 'smooth_intercept',
                              'subsmooth': 'subsmooth_intercept'})
        for cov in self.covariates:
            self.mulcov_table.append({'covariate': cov[0]['name'], 'type': cov[0]['type'],
                                      'effected': cov[0]['effected'], 'group': 'all',
                                      'smooth': 'smooth_mulcov_' + cov[0]['name']})

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
                row['subgroup'] = self.data.loc[data_id, 'group']
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
                    row[cov[0]['name']] = self.data.loc[data_id, cov[0]['name']]
                self.data_table.append(copy.copy(row))

    def create_avgint_table(self):

        rate_to_integrand = {'iota': 'Sincidence', 'rho': 'remission', 'chi': 'mtexcess', 'omega': 'mtother'}

        used = set()

        self.avgint_table = []
        row = {'intercept': 1.0, 'weight': 'constant', 'gamma_one': 1.0}
        row.update({cov[0]['name']: 0.0 for cov in self.covariates})
        for rate in self.rates:
            for age in self.age_list:
                for time in self.time_list:
                    for group in self.groups:
                        for loc in self.location_names:
                            row['integrand'] = rate_to_integrand[rate]
                            used.add(rate_to_integrand[rate])
                            row['node'] = loc
                            row['subgroup'] = group
                            row['age_lower'] = age
                            row['age_upper'] = age
                            row['time_lower'] = time
                            row['time_upper'] = time
                            self.avgint_table.append(copy.copy(row))
                        row['node'] = 'all'
                        self.avgint_table.append(copy.copy(row))

        #for i in range(len(self.covariates)):
        #    assert self.mulcov_table[i+2]['covariate'] == self.covariates[i]['name']
        #    self.integrand.append('mulcov_' + str(i+2))

        for integrand in self.integrand:
            if integrand not in used:
                for age in self.age_list:
                    for time in self.time_list:
                        for group in self.groups:
                            for loc in self.location_names:
                                row['integrand'] = integrand
                                row['node'] = loc
                                row['subgroup'] = group
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

    def create_smooth_table(self, child_grid: bool = True):
        self.smooth_table = [{'name': 'smooth_gamma_one',
                              'age_id': [int(len(self.age_list)/2)],
                              'time_id': [int(len(self.time_list)/2)],
                              'fun': lambda a, t: ('prior_gamma_one', None, None)},
                             {'name': 'smooth_intercept',
                              'age_id': [int(len(self.age_list)/2)],
                              'time_id': [int(len(self.time_list)/2)],
                              'fun': lambda a, t: ('prior_intercept', None, None)},
                             {'name': 'subsmooth_intercept',
                              'age_id': [int(len(self.age_list) / 2)],
                              'time_id': [int(len(self.time_list) / 2)],
                              'fun': lambda a, t: ('prior_intercept_sub', None, None)},
                             ]
        for rate in self.rates:
            self.smooth_table.append({'name': 'smooth_rate_' + rate,
                                      'age_id': range(len(self.age_list)), 'time_id': range(len(self.time_list)),
                                      'fun': lambda a, t, r=rate: ('value_prior_' + r,
                                                                   'dage_prior_' + r, 'dtime_prior_' + r)})
            # use sparse grid for child random effects
            if child_grid == "one":
                self.smooth_table.append({'name': 'smooth_rate_child_' + rate,
                                          'age_id': [len(self.age_list)//2],
                                          'time_id': [len(self.time_list)//2],
                                          'fun': lambda a, t, r=rate: ('value_prior_child_' + r, None, None)})
            elif child_grid == "two":
                self.smooth_table.append({'name': 'smooth_rate_child_' + rate,
                                          'age_id': [0, len(self.age_list) - 1], 'time_id': [0, len(self.time_list) - 1],
                                          'fun': lambda a, t, r=rate: ('value_prior_child_' + r,
                                                                       'dage_prior_child_' + r,
                                                                       'dtime_prior_child_' + r)})
            else:
                self.smooth_table.append({'name': 'smooth_rate_child_' + rate,
                                          'age_id': range(len(self.age_list)),
                                          'time_id': range(len(self.time_list)),
                                          'fun': lambda a, t, r=rate: ('value_prior_child_' + r,
                                                                       'dage_prior_child_' + r,
                                                                       'dtime_prior_child_' + r)})

        for cov in self.covariates:
            name = cov[0]['name']
            age_min = -float('inf')
            age_max = float('inf')
            if cov[1] is not None:
                age_min = cov[1][0]
                age_max = cov[1][1]

            def fun(a, t, na):
                if age_min - .5 <= a <= age_min + .5 or age_max -.5 <= a <= age_max + .5:
                    return ('value_prior_' + na, 'prior_zero', None)
                else:
                    return (0.0, None, None)

            self.smooth_table.append({'name': 'smooth_mulcov_' + cov[0]['name'],
                                      'age_id': range(len(self.age_list)), 'time_id': [len(self.time_list)//2],
                                      'fun': lambda a, t: fun(a, t, name)})

    def create_prior_table(self):
        self.prior_table = [{'name': 'prior_gamma_one', 'density': 'uniform',
                             'lower': 0.0, 'mean': 0.0, 'upper': 0.0},
                            {'name': 'prior_intercept', 'density': 'uniform',
                             'lower': 0.0, 'mean': 0.0, 'upper': 0.0}]
        self.prior_table.append(self.subgroup_value_prior)

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
            if self.rate_child_priors[i][1] is not None:
                self.prior_table[-1].update(self.rate_child_priors[i][1])
            else:
                self.prior_table[-1].update({'density': 'uniform', 'mean': 0.0, 'upper': 0.0, 'lower': 0.0})
            self.prior_table.append({'name': 'dtime_prior_child_' + self.rates[i]})
            if self.rate_child_priors[i][2] is not None:
                self.prior_table[-1].update(self.rate_child_priors[i][2])
            else:
                self.prior_table[-1].update({'density': 'uniform', 'mean': 0.0, 'upper': 0.0, 'lower': 0.0})

        for i in range(len(self.covariates)):
            self.prior_table.append({'name': 'value_prior_' + self.covariates[i][0]['name']})
            self.prior_table[-1].update(self.cov_priors[i])
            #self.prior_table.append({'name': 'dage_prior_' + self.covariates[i]['name']})
            #self.prior_table[-1].update(self.cov_priors[i][1])
            #self.prior_table.append({'name': 'dtime_prior_' + self.covariates[i]['name']})
            #self.prior_table[-1].update(self.cov_priors[i][2])

        self.prior_table.append({'name': 'prior_zero', 'density': 'gaussian',
                                 'upper': 0.0, 'lower': 0.0, 'mean': 0.0, 'std': 1e-10})

    def create_option_table(self):
        self.option_table = [
            {'name': 'parent_node_name', 'value': 'all'},
            {'name': 'ode_step_size', 'value': '5.0'},
            {'name': 'quasi_fixed', 'value': 'false'},
            {'name': 'max_num_iter_fixed', 'value': '200'},
            {'name': 'print_level_fixed', 'value': '5'},
            {'name': 'tolerance_fixed', 'value': '1e-4'},
            {'name': 'meas_noise_effect', 'value': 'add_var_scale_all'},
            {'name': 'zero_sum_mulcov_group', 'value': 'all'}
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

        rate_to_integrand = {'iota': 'Sincidence', 'rho': 'remission', 'chi': 'mtexcess', 'omega': 'mtother'}
        for rate in self.rates:
            if rate_to_integrand[rate] not in self.integrand:
                self.integrand_table.append({'name': rate_to_integrand[rate]})

        self.node_table = [{'name': 'all', 'parent': ''}]
        for loc in self.location_names:
            self.node_table.append({'name': loc, 'parent': 'all'})

        self.subgroup_table = [{'subgroup': 'all', 'group': 'all'}]
        if len(self.groups) > 1:
            self.subgroup_table = [{'subgroup': group, 'group': 'all'} for group in self.groups]

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
            self.option_table.append({'name': 'zero_sum_child_rate', 'value': ' '.join(self.rates)})
            n = len(self.option_name_id)
            self.option_name_id['zero_sum_random'] = n

    def remove_zero_sum_constraint(self):
        if 'zero_sum_random' in self.option_name_id:
            self.option_table[self.option_name_id['zero_sum_child_rate']]['value'] = ''


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
            self.subgroup_table,
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

        if not os.path.exists(self.path):
            os.mknod(self.path)

        command = [program, self.path, 'init']
        print(' '.join(command))
        flag = subprocess.call(command)
        if flag != 0:
            sys.exit('The dismod_at init command failed')
        if db2csv is True:
            dismod_at.db2csv_command(self.path)
