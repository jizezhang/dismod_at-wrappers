import numpy as np
import sys
import subprocess
import copy
import pandas as pd
from typing import Dict, List, Any, Tuple
import dismod_at

program = '/home/prefix/dismod_at.release/bin/dismod_at'


def system_command(command, verbose=True):
    if verbose:
        print(' '.join(command[1:]))
    flag = subprocess.call(command)
    if flag != 0:
        sys.exit('command failed: flag = ' + str(flag))
    return


class TwoLevel:

    def __init__(self, data: pd.DataFrame, location_names: List[str],
                 integrand: List[str], rates: List[str],
                 rate_parent_priors: List[Tuple[Dict[str, str], ...]],
                 rate_child_priors: List[Tuple[Dict[str, str], ...]],
                 meas_noise_density: Dict[str, Dict[str, Any]],
                 path_to_db: str,
                 covariates: List[Dict[str, str]] = None,
                 cov_priors: List[Tuple[Dict[str, str], ...]] = None,
                 age_list: List[int] = None, time_list: List[int] = None,
                 options: List[Dict[str, str]] = None):

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
        if options is not None:
            self.options = options
        else:
            self.options = []
        self.path = path_to_db
        if age_list is not None:
            self.age_list = age_list
        else:
            self.age_list = []
        if time_list is not None:
            self.time_list = time_list
        else:
            self.time_list = []

        self.zero_sum = False

        self.check()

    def check(self):
        assert all([x in self.meas_noise_density for x in self.integrand])
        assert self.m == len(self.cov_priors)
        assert len(self.rates) == len(self.rate_parent_priors)
        assert (len(self.rates) == len(self.rate_child_priors) or len(self.rate_child_priors) == 1)

    def init_database(self, use_gamma: bool = False, use_lambda: bool = True, max_iter: int = 100, tol: float = 1e-4,
                      zero_sum: bool = False):
        if len(self.age_list) == 0:
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
            self.age_list = age_list
        if len(self.time_list) == 0:
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
            self.time_list = time_list

        avgint_table = list()
        nslist_table = dict()  # smoothing
        integrand_table = []
        for intg in self.integrand:
            integrand_table.append({'name': intg})

        self.node_table = [{'name': 'world', 'parent': ''}]
        for loc in self.location_names:
            self.node_table.append({'name': loc, 'parent': 'world'})

        weight_table = [{'name': 'constant', 'age_id': range(len(self.age_list)),
                         'time_id': range(len(self.time_list)), 'fun': lambda a, t: 1.0}]

        rate_table = list()
        for rate in self.rates:
            rate_table.append({'name': rate,
                               'parent_smooth': 'smooth_rate_' + rate,
                               'child_smooth': 'smooth_rate_child_' + rate})

        covariate_table = []
        if use_gamma is True:
            covariate_table.append({'name': 'one', 'reference': 0.0})
        for cov in self.covariates:
            covariate_table.append({'name': cov['name'], 'reference': 0.0})
        mulcov_table = []
        if use_gamma is True:
            mulcov_table.append({'covariate': 'one', 'type': 'meas_noise', 'effected': 'Sincidence',
                                 'smooth': 'smooth_gamma_one'})
        for cov in self.covariates:
            mulcov_table.append({'covariate': cov['name'], 'type': cov['type'],
                                 'effected': cov['effected'], 'smooth': 'smooth_mulcov_' + cov['name']})

        smooth_table = list()
        for rate in self.rates:
            smooth_table.append({'name': 'smooth_rate_' + rate,
                                 'age_id': range(len(self.age_list)), 'time_id': range(len(self.time_list)),
                                 'fun': lambda a, t, r=rate: (
                                 'value_prior_' + r, 'dage_prior_' + r, 'dtime_prior_' + r)})
            smooth_table.append({'name': 'smooth_rate_child_' + rate,
                                 'age_id': range(len(self.age_list)), 'time_id': range(len(self.time_list)),
                                 'fun': lambda a, t, r=rate: ('value_prior_child_' + r,
                                                              'dage_prior_child_' + r, 'dtime_prior_child_' + r),
                                 'mulstd_value_prior_name': 'prior_lambda'})
        if use_gamma is True:
            smooth_table.append({'name': 'smooth_gamma_one',
                                 'age_id': range(len(self.age_list)), 'time_id': range(len(self.time_list)),
                                 'fun': lambda a, t: ('prior_uniform', 'prior_zero', 'prior_zero')})
        for cov in self.covariates:
            name = cov['name']
            smooth_table.append({'name': 'smooth_mulcov_' + cov['name'],
                                 'age_id': range(len(self.age_list)), 'time_id': range(len(self.time_list)),
                                 'fun': lambda a, t, name=name: ('value_prior_' + name, 'dage_prior_' + name,
                                                                 'dtime_prior_' + name)})

        self.prior_table = [{'name': 'prior_uniform', 'density': 'uniform', 'lower': 0.0, 'mean': 0.0, 'upper': 10.},
                            {'name': 'prior_zero', 'density': 'uniform', 'lower': 0.0, 'mean': 0.0, 'upper': 0.0}]
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

        if use_lambda is True:
            self.prior_table.append({'name': 'prior_lambda', 'density': 'uniform',
                                     'mean': 1.0, 'lower': 0.0, 'upper': 100})
        else:
            self.prior_table.append({'name': 'prior_lambda', 'density': 'uniform',
                                     'mean': 1.0, 'lower': 1.0, 'upper': 1.})

        for i in range(len(self.covariates)):
            self.prior_table.append({'name': 'value_prior_' + self.covariates[i]['name']})
            self.prior_table[-1].update(self.cov_priors[i][0])
            self.prior_table.append({'name': 'dage_prior_' + self.covariates[i]['name']})
            self.prior_table[-1].update(self.cov_priors[i][1])
            self.prior_table.append({'name': 'dtime_prior_' + self.covariates[i]['name']})
            self.prior_table[-1].update(self.cov_priors[i][2])

        self.data_table = list()
        row = {
            'weight': 'constant',
            'hold_out': False,
            'one': 1.0
        }
        row.update(self.meas_noise_density)
        for data_id in range(self.n):
            if self.data.loc[data_id, 'measure'] in self.integrand and \
                    self.data.loc[data_id, 'location_name'] in self.location_names:
                row['node'] = self.data.loc[data_id, 'location_name']
                row['integrand'] = self.data.loc[data_id, 'measure']
                for k, v in self.meas_noise_density[row['integrand']].items():
                    row[k] = v
                row['meas_value'] = self.data.loc[data_id, 'mean']
                row['meas_std'] = self.data.loc[data_id, 'standard_error']
                row['age_lower'] = self.data.loc[data_id, 'age_start']
                row['age_upper'] = self.data.loc[data_id, 'age_end']
                row['time_lower'] = self.data.loc[data_id, 'year_start']
                row['time_upper'] = self.data.loc[data_id, 'year_end']
                for cov in self.covariates:
                    row[cov['name']] = self.data.loc[data_id, cov['name']]
                self.data_table.append(copy.copy(row))

        self.option_table = [
            {'name': 'parent_node_name', 'value': 'world'},
            {'name': 'ode_step_size', 'value': '10.0'},
            {'name': 'quasi_fixed', 'value': 'false'},
            {'name': 'max_num_iter_fixed', 'value': max_iter},
            {'name': 'print_level_fixed', 'value': '5'},
            {'name': 'tolerance_fixed', 'value': str(tol)},
            {'name': 'meas_noise_effect', 'value': 'add_var_scale_all'},
            #{'name': 'zero_sum_random', 'value': 'iota'},
        ]

        if zero_sum and not self.zero_sum:
            self.option_table.append({'name': 'zero_sum_random', 'value': 'iota'})
            self.zero_sum = True

        if self.integrand == ['Sincidence']:
            self.option_table.append({'name': 'rate_case', 'value': 'iota_pos_rho_zero'})
        elif self.integrand == ['remission']:
            self.option_table.append({'name': 'rate_case', 'value': 'iota_zero_rho_pos'})
        else:
            self.option_table.append({'name': 'rate_case', 'value': 'iota_pos_rho_pos'})

        option_name_id = {}
        for i in range(len(self.option_table)):
            option_name_id[self.option_table[i]['name']] = i
        for option in self.options:
            if option['name'] in option_name_id:
                self.option_table[option_name_id[option['name']]]['value'] = option['value']
            else:
                self.option_table.append(option)

        dismod_at.create_database(
            self.path,
            self.age_list,
            self.time_list,
            integrand_table,
            self.node_table,
            weight_table,
            covariate_table,
            avgint_table,
            self.data_table,
            self.prior_table,
            smooth_table,
            nslist_table,
            rate_table,
            mulcov_table,
            self.option_table
        )

    def initialize(self, db2csv=False):
        command = [program, self.path, 'init']
        print(' '.join(command))
        flag = subprocess.call(command)
        if flag != 0:
            sys.exit('The dismod_at init command failed')
        if db2csv is True:
            dismod_at.db2csv_command(self.path)

    def fit_fixed(self, tol: float = 1e-4, use_gamma: bool = False, use_lambda: bool = False,
                  db2csv: bool = True, max_iter: int = 100, zero_sum: bool = False):
        self.init_database(use_gamma=use_gamma, use_lambda=use_lambda, tol=tol, max_iter=max_iter, zero_sum=zero_sum)
        self.initialize()
        system_command([program, self.path, 'fit', 'fixed'])
        if db2csv:
            dismod_at.db2csv_command(self.path)

    def fit_both(self, use_gamma: bool = False, use_lambda: bool = False, tol: float = 1e-4,
                 fit_fixed: bool = True, db2csv: bool = True, max_iter: int = 100, fit_gaussian: bool = False,
                 zero_sum: bool = False):

        if fit_gaussian:
            for i, row in enumerate(self.data_table):
                self.data_table[i]['density'] = 'gaussian'

        if fit_fixed:
            self.fit_fixed(use_gamma=use_gamma, use_lambda=use_lambda,
                           tol=tol, db2csv=False, max_iter=max_iter, zero_sum=zero_sum)
            system_command([program, self.path, 'set', 'start_var', 'fit_var'])
        else:
            self.init_database(use_gamma=use_gamma, use_lambda=use_lambda, tol=tol, max_iter=max_iter,
                               zero_sum=zero_sum)
            self.initialize()

        if fit_gaussian:
            for i, row in enumerate(self.data_table):
                self.data_table[i].update(self.meas_noise_density)

        command = [program, self.path, 'fit', 'both']
        print(' '.join(command[1:]))
        flag = subprocess.call(command)
        if flag != 0:
            sys.exit('The dismod_at fit both command failed')
        if db2csv:
            dismod_at.db2csv_command(self.path)

    # def updateMeasNoiseDensity(self, density_name: str,
    #                            params: Dict[str, Any]):
    #     connection = dismod_at.create_connection(self.path, False)
    #     density_table = dismod_at.get_table_dict(connection, 'density')
    #     command = 'UPDATE data SET density_id = ' \
    #         + str(self.density_dict[density_name])
    #     print(command)
    #     dismod_at.sql_command(connection, command)
    #     for k, v in params.items():
    #         command = 'UPDATE data SET ' + k + ' = ' + str(v)
    #         dismod_at.sql_command(connection, command)
    #         print(command)
    #     connection.close()
