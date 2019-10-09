import numpy as np
import sys
import subprocess
import copy
from dismod_db import DismodDB
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

    def __init__(self, db: DismodDB):

        """
        """
        self.db = db
        self.db_path = db.path

    def initialize(self, db2csv=False):
        self.db.init_database(db2csv)

    def fit_fixed(self, tol: float = 1e-4, db2csv: bool = True, max_iter: int = 100, zero_sum: bool = False,
                  print_level: int = 5):
        self.db.set_tol(tol)
        self.db.set_max_iteration(max_iter)
        self.db.set_print_level(print_level)
        if zero_sum:
            self.db.set_zero_sum_constraint()
        self.initialize(db2csv=False)
        system_command([program, self.db_path, 'fit', 'fixed'])
        system_command([program, self.db_path, 'predict', 'fit_var'])
        if db2csv:
            dismod_at.db2csv_command(self.db_path)

    def fit_both(self, tol: float = 1e-4, fit_fixed: bool = True, db2csv: bool = True, max_iter: int = 100,
                 fit_gaussian: bool = False, zero_sum: bool = False, print_level: int = 5):

        if fit_gaussian:
            self.db.set_meas_density('gaussian')

        if fit_fixed:
            self.fit_fixed(tol=tol, db2csv=False, max_iter=max_iter, zero_sum=zero_sum, print_level=print_level)
            system_command([program, self.db_path, 'set', 'start_var', 'fit_var'])
        else:
            self.db.set_tol(tol)
            self.db.set_max_iteration(max_iter)
            self.db.set_print_level(print_level)
            if zero_sum:
                self.db.set_zero_sum_constraint()
            self.initialize(db2csv=False)

        if fit_gaussian:
            self.db.reset_meas_density()
            self.initialize(db2csv=False)

        system_command([program, self.db_path, 'fit', 'both'])
        system_command([program, self.db_path, 'predict', 'fit_var'])

        if db2csv:
            dismod_at.db2csv_command(self.db_path)


