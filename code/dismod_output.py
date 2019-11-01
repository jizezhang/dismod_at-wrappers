import numpy as np
import sys
import subprocess
import pandas as pd
from typing import Dict, List, Any, Tuple
import dismod_at
import sqlite3
import shutil
from sqlalchemy import create_engine
import pymysql

program = '/home/prefix/dismod_at.release/bin/dismod_at'

age_intervals = [(0, 7./365), (7./365, 28./365), (28./365, 1)]+[(0, 1), (1, 5)] + \
                [(x, x+5) for x in np.arange(5, 100, 5)]
age_group_ids = [2, 3, 4] + [28, 5] + list(range(6, 21)) + list(range(30, 34))
assert len(age_intervals) == len(age_group_ids)
n_age = len(age_group_ids)
age_id_to_range = {age_group_ids[i]: age_intervals[i] for i in range(n_age)}

integrand_to_measure_id = {'prevalence': 5, 'remission': 7, 'mtexcess': 9, 'relrisk': 11, 'mtstandard': 12,
                           'mtwith': 13, 'mtall': 14, 'mtspecific': 15, 'mtother': 16, 'Sincidence': 41,
                           'susceptible': 39, 'withC': 40, 'Tincidence': 42}

def system_command(command, verbose=True):
    if verbose:
        print(' '.join(command[1:]))
    flag = subprocess.call(command)
    if flag != 0:
        sys.exit('command failed: flag = ' + str(flag))
    return


class DismodOutput:

    def __init__(self, path_to_db: str):
        self.path_to_db = path_to_db
        self.node_id_to_loc = self.get_node_names()
        self.age_list, self.time_list = self.get_age_time_lists()
        self.age_min, self.age_max = min(self.age_list), max(self.age_list)
        self.time_min, self.time_max = min(self.time_list), max(self.time_list)

    def get_age_time_lists(self):
        conn = sqlite3.connect(self.path_to_db)
        age = pd.read_sql_query("select age from age;", conn)
        time = pd.read_sql_query("select time from time", conn)
        conn.close()
        return age.values.squeeze(), time.values.squeeze()

    def get_node_names(self):
        conn = sqlite3.connect(self.path_to_db)
        df = pd.read_sql_query("select node_id, node_name from node;", conn)
        node_id_to_name = {}
        for i, row in df.iterrows():
            node_id_to_name[row['node_id']] = row['node_name']
        print(node_id_to_name)
        return node_id_to_name

    def get_covarates_names(self):
        conn = sqlite3.connect(self.path_to_db)
        df = pd.read_sql_query("select covariate.covariate_id, covariate_name, mulcov_type \
                                from covariate \
                                inner join mulcov on covariate.covariate_id == mulcov.covariate_id;", conn)
        cov_name_to_id = {}
        for i, row in df.iterrows():
            if row['mulcov_type'] == 'rate_value':
                cov_name_to_id[row['covariate_name']] = 'mulcov_' + str(row['covariate_id'])
        conn.close()

        return cov_name_to_id

    def get_covariate_multiplier_values(self, cov_names: List[str] = None):
        conn = sqlite3.connect(self.path_to_db)
        df = pd.read_sql_query("select covariate_name, fit_var_value \
                               from covariate inner join mulcov on covariate.covariate_id == mulcov.covariate_id \
                               left join var on mulcov.covariate_id == var.covariate_id \
                               left join fit_var on fit_var.fit_var_id == var.var_id;", conn)
        conn.close()
        if cov_names is not None:
            return df[df['covariate_name'].isin(cov_names)]
        return df

    def get_integrand_values(self, path_to_db=None):
        if path_to_db is None:
            conn = sqlite3.connect(self.path_to_db)
        else:
            conn = sqlite3.connect(path_to_db)
        df = pd.read_sql_query("select node.node_name, integrand.integrand_name, avgint.*, avg_integrand \
                                from avgint \
                                inner join predict on avgint.avgint_id == predict.avgint_id \
                                inner join node on node.node_id == avgint.node_id \
                                inner join integrand on avgint.integrand_id == integrand.integrand_id;", conn)
        conn.close()
        return df

    def create_GBD_integrand(self, integrands: List[str], time_list: List[int], sex_ids: List[int],
                             location_name_to_id: Dict[str, int]):
        path = self.path_to_db[:-3] + '_gbd.db'
        print(path)
        shutil.copyfile(self.path_to_db, path)
        connection = sqlite3.connect(path)
        crsr = connection.cursor()
        crsr.execute('select count(covariate_id) from covariate')
        n_covs = crsr.fetchall()[0][0]
        crsr.execute("drop table integrand")
        row_list = []
        for name in integrands:
            row_list.append([0.0, name])
        print(row_list)
        dismod_at.create_table(connection, 'integrand',
                               ['minimum_meas_cv', 'integrand_name'],
                               ['real', 'text'], row_list)

        crsr.execute("drop table avgint")

        row_list = []
        for integrand_id in range(len(integrands)):
            for age_id in age_group_ids:
                for time in time_list:
                    for node_id, node_name in self.node_id_to_loc.items():
                        if node_name in location_name_to_id:
                            age_lower = age_id_to_range[age_id][0]
                            age_upper = age_id_to_range[age_id][1]
                            if age_lower >= self.age_min and age_upper <= self.age_max and \
                                    self.time_min <= time <= self.time_max:
                                row = [integrand_id, node_id, None, age_lower, age_upper, time, time]
                                row.extend([None]*n_covs)
                                row.extend([location_name_to_id[node_name], age_id, time,
                                            integrand_to_measure_id[integrands[integrand_id]]])
                                for sex_id in sex_ids:
                                    row_list.append(row + [sex_id])

        dismod_at.create_table(connection, 'avgint', ['integrand_id', 'node_id', 'weight_id', 'age_lower', 'age_upper',
                                                      'time_lower', 'time_upper'] +
                               ['x_' + str(i) for i in range(n_covs)] +
                               ['location_id', 'age_group_id', 'year_id', 'measure_id', 'sex_id'],
                               ['integer', 'integer', 'integer', 'real', 'real', 'real', 'real'] + ['real']*n_covs + \
                               ['integer']*5, row_list)
        connection.close()
        system_command([program, path, 'predict', 'fit_var'])

    def save_GBD_output(self, integrands: List[str], model_version_id: int, time_list: List[int], sex_ids: List[int], location_name_to_id: Dict[str, int],
                        path_to_csv: str):
        self.create_GBD_integrand(integrands, time_list, sex_ids, location_name_to_id)
        df = self.get_integrand_values(self.path_to_db[:-3] + '_gbd.db')
        df.rename(columns={'avg_integrand': 'mean'}, inplace=True)
        df['lower'] = df['mean']  # dummy fill-in for now
        df['upper'] = df['mean']
        df['model_version_id'] = model_version_id
        gbd_output = df[['model_version_id', 'location_id', 'age_group_id', 'sex_id', 'year_id', 'measure_id', 'mean',
                         'lower', 'upper']]
        gbd_output.reset_index(drop=True, inplace=True)
        gbd_output.to_csv(path_to_csv, index=False)

        # ----- write to database ---------
        # engine = create_engine('mysql+pymysql://jizez:jizez100@epidecomp-perconavm-db-d01.db.ihme.washington.edu/epi',
        #                        echo=False, pool_timeout=60)
        # df.to_sql('model_estimate_final', con=engine, if_exists='append', index=False, chunksize=50, method='multi')
        connection = pymysql.connect(user='jizez', password='jizez100',
                                     host='epidecomp-perconavm-db-d01.db.ihme.washington.edu', database='epi')
        cursor = connection.cursor()
        cursor.executemany("insert ignore into model_estimate_final " +
                           "(model_version_id, location_id, age_group_id, sex_id, year_id, measure_id, mean, lower, upper) " +
                           "values (%s, %s, %s, %s, %s, %s, %s, %s, %s) ",
                           list(gbd_output.itertuples(index=False, name=None)))
        connection.commit()
        connection.close()
