import numpy as np
import sys
import os
import distutils.dir_util
import subprocess
import copy
import math
import random
import statistics
import pandas as pd
from typing import Dict, List, Any, Tuple
import dismod_at
program = '/home/prefix/dismod_at.release/bin/dismod_at'


class FixedOnly:

    def __init__(self, data: pd.DataFrame, integrand: List[str], rates: List[str],
                 rate_priors: List[Tuple[Dict[str, str],...]],
                 meas_noise_density:Dict[str, Dict[str,Any]],
                 path_to_db: str,
                 covariates: List[Dict[str,str]]=[],
                 cov_priors: List[Tuple[Dict[str, str],...]]=[],
                 age_list=[], time_list=[], options=[],max_iter=500):

        """
        multiple integrands
        """

        self.density_dict = {'uniform': 0, 'gaussian': 1, 'laplace': 2,
                             'students': 3, 'log_gaussian': 4,
                             'log_laplace': 5, 'log_students': 6}
        self.integrand = integrand
        self.meas_noise_density = meas_noise_density
        self.rates = rates
        self.data = data
        self.n = self.data.shape[0]
        self.m = len(covariates)
        self.covariates = covariates
        self.rate_priors = rate_priors
        self.cov_priors = cov_priors
        self.options = options
        self.path = path_to_db
        self.age_list = age_list
        self.time_list = time_list

        self.check()
        self.initDatabase(max_iter=max_iter)

    def check(self):
        assert all([x in self.meas_noise_density for x in self.integrand])
        assert self.m == len(self.cov_priors)
        assert len(self.rates) == len(self.rate_priors)

    def initDatabase(self, max_iter=500):
        if len(self.age_list) == 0:
            max_age = -float('inf')
            min_age = float('inf')
            for i in range(self.n):
                max_age = max(max_age,self.data.loc[i,'age_end'])
                min_age = min(min_age, self.data.loc[i,'age_start'])
            age_list = [int(round(x)) for x in np.linspace(min_age,max_age,\
                        round((max_age-min_age)/5)+1)]
            age_list = sorted(list(set(age_list)))
            #if len(age_list) == 1:
            #    age_list.insert(0,age_list[0]-1)
            self.age_list = age_list
        if len(self.time_list) == 0:
            max_time = -float('inf')
            min_time = float('inf')
            for i in range(self.n):
                max_time = max(max_time, self.data.loc[i,'year_end'])
                min_time = min(min_time, self.data.loc[i,'year_start'])
            time_list = [int(round(x)) for x in np.linspace(min_time, max_time,\
                         round((max_time - min_time)/3+1))]
            time_list = sorted(list(set(time_list)))
            self.time_list = time_list
        #print(self.age_list)
        #print(self.time_list)

        avgint_table = list()
        nslist_table = dict() # smoothing
        integrand_table = []
        for intg in self.integrand:
            integrand_table.append({'name': intg})

        node_table = [{'name':'world', 'parent':''}]

        weight_table = [{ 'name': 'constant',  'age_id': range(len(self.age_list)),
                        'time_id': range(len(self.time_list)), 'fun': lambda a,t: 1.0 }]

        rate_table = list()
        for rate in self.rates:
            rate_table.append({'name': rate,
                      'parent_smooth': 'smooth_rate_'+rate})

        covariate_table = list()
        for cov in self.covariates:
            covariate_table.append({'name':cov['name'],'reference':0.0})
        mulcov_table = list()
        for cov in self.covariates:
            mulcov_table.append({'covariate':cov['name'], 'type':cov['type'],
                                 'effected': cov['effected'], 'smooth':'smooth_mulcov_'+cov['name']})

        smooth_table = list()
        for rate in self.rates:
            smooth_table.append({'name':'smooth_rate_'+rate,
                                 'age_id':range(len(self.age_list)),'time_id':range(len(self.time_list)),
                                 'fun': lambda a,t,r=rate:('value_prior_'+r,'dage_prior_'+r,'dtime_prior_'+r)})
        for cov in self.covariates:
            name = cov['name']
            smooth_table.append({'name':'smooth_mulcov_'+cov['name'],
                                 'age_id':range(len(self.age_list)),'time_id':range(len(self.time_list)),
                                 'fun': lambda a,t,name=name:('value_prior_'+name,'dage_prior_'+name,'dtime_prior_'+name)})
        #for row in smooth_table:
        #    print(row['fun'](0,0))

        prior_table = []
        for i in range(len(self.rates)):
            prior_table.append({'name': 'value_prior_'+self.rates[i]})
            prior_table[-1].update(self.rate_priors[i][0])
            prior_table.append({'name': 'dage_prior_'+self.rates[i]})
            prior_table[-1].update(self.rate_priors[i][1])
            prior_table.append({'name': 'dtime_prior_'+self.rates[i]})
            prior_table[-1].update(self.rate_priors[i][2])
        for i in range(len(self.covariates)):
            prior_table.append({'name': 'value_prior_'+self.covariates[i]['name']})
            prior_table[-1].update(self.cov_priors[i][0])
            prior_table.append({'name': 'dage_prior_'+self.covariates[i]['name']})
            prior_table[-1].update(self.cov_priors[i][1])
            prior_table.append({'name': 'dtime_prior_'+self.covariates[i]['name']})
            prior_table[-1].update(self.cov_priors[i][2])

        data_table = list()
        row = {
            'node': 'world',
            'weight':      'constant',
            'hold_out':     False,
        }
        row.update(self.meas_noise_density)
        for data_id in range(self.n):
            if self.data.loc[data_id, 'measure'] in self.integrand:
                row['integrand'] = self.data.loc[data_id, 'measure']
                for k, v in self.meas_noise_density[row['integrand']].items():
                    row[k] = v
                row['meas_value']  = self.data.loc[data_id,'meas_value']
                row['meas_std'] = self.data.loc[data_id,'meas_std']
                row['age_lower'] = self.data.loc[data_id,'age_start']
                row['age_upper'] = self.data.loc[data_id,'age_end']
                row['time_lower'] = self.data.loc[data_id,'year_start']
                row['time_upper'] = self.data.loc[data_id,'year_end']
                for cov in self.covariates:
                    row[cov['name']] = self.data.loc[data_id,cov['name']]
                data_table.append( copy.copy(row) )

        option_table = [
              { 'name':'parent_node_name',       'value':'world'     },
              { 'name':'ode_step_size',          'value':'10.0'              },
              { 'name':'quasi_fixed',            'value':'false'             },
              { 'name':'max_num_iter_fixed',     'value':max_iter               },
              { 'name':'print_level_fixed',      'value':'5'                 },
              { 'name':'tolerance_fixed',        'value':'1e-8'             },
	     ]
        if self.integrand == ['Sincidence']:
            option_table.append({'name':'rate_case','value':'iota_pos_rho_zero'})
        elif self.integrand == 'remission':
            option_table.append({'name':'rate_case','value':'iota_zero_rho_pos'})
        else:
            option_table.append({'name':'rate_case','value':'iota_pos_rho_pos'})

        option_name_id = {}
        for i in range(len(option_table)):
            option_name_id[option_table[i]['name']] = i
        for option in self.options:
            if option['name'] in option_name_id:
                option_table[option_name_id[option['name']]]['value'] = option['value']
            else:
                option_table.append(option)


        dismod_at.create_database(
		self.path,
		self.age_list,
		self.time_list,
		integrand_table,
		node_table,
		weight_table,
		covariate_table,
		avgint_table,
		data_table,
		prior_table,
		smooth_table,
		nslist_table,
		rate_table,
		mulcov_table,
		option_table
        )

        command = [program, self.path, 'init']
        print(' '.join(command))
        flag = subprocess.call(command)
        if flag != 0:
            sys.exit('The dismod_at init command failed')
        dismod_at.db2csv_command(self.path)

    def fit_fixed(self, db2csv=True):
        command = [program, self.path, 'fit', 'fixed']
        print(' '.join(command))
        flag = subprocess.call(command)
        if flag != 0:
            sys.exit('The dismod_at fit fixed command failed')
        if db2csv:
            dismod_at.db2csv_command(self.path)

    def updateMeasNoiseDensity(self, density_name: str,
                               params: Dict[str, Any]):
        connection = dismod_at.create_connection(self.path, False)
        density_table = dismod_at.get_table_dict(connection, 'density')
        command = 'UPDATE data SET density_id = ' \
            + str(self.density_dict[density_name])
        print(command)
        dismod_at.sql_command(connection, command)
        for k, v in params.items():
            command = 'UPDATE data SET ' + k + ' = ' + str(v)
            dismod_at.sql_command(connection, command)
            print(command)
        connection.close()
