import numpy as np
import time
import warnings
warnings.simplefilter('error')

age_intervals = [(0, 7./365), (7./365, 28./365), (28./365, 1)]+[(0, 1), (1, 5)] + \
                [(x, x+5) for x in np.arange(5, 115, 5)] + [(100, 125)] + [(0, 125)]
age_group_ids = [2, 3, 4] + [28, 5] + list(range(6, 21)) + list(range(30, 34)) + list(range(44, 47)) + [48, 22]
assert len(age_intervals) == len(age_group_ids)
n_age = len(age_group_ids)
age_id_to_range = {age_group_ids[i]: age_intervals[i] for i in range(n_age)}


def get_age_year_value(cov, loc_ids, sex_ids):
    """
    Build a dictionary where key is (loc_id, age_id) and value is
    a list of (age_group_id, year_id, mean_value) collected from
    covariate data. If covariate from a certain location is not
    available, the list is empty.

    Args:
        cov (pandas.Dataframe): a dataframe storing covariate info
        loc_ids (list[int]): location ids
        sex_ids (list[int]): sex ids

    Returns:
        dct (dict[tuple(int, int), list[tuple(int, int, float)]])

    """
    dct = {}
    avail_locs = set(cov.location_id.unique()) & set(loc_ids)
    if len(avail_locs) == 0:
        print("covariate file does not contain locations", loc_ids, " requested")
    cov = cov[(cov['location_id'].isin(list(avail_locs))) & (cov['age_group_id'].isin(age_group_ids))]
    for loc_id in loc_ids:
        for sex_id in sex_ids:
            if sex_id in set(cov['sex_id'].values):
                cov_sub = cov[(cov['location_id'] == loc_id) & (cov['sex_id'] == sex_id)]
                cov_sub = cov_sub.sort_values(['age_group_id', 'year_id'])
                dct[(loc_id, sex_id)] = list(cov_sub[['age_group_id', 'year_id', 'mean_value']].values)
            else:
                # either sex_id = 3, in which case aggregate sex_id = 1, 2, or sex_id = 1 or 2, and all cov has sex_id = 3
                cov_sub = cov[cov['location_id'] == loc_id]
                cov_sub = cov_sub.sort_values(['age_group_id', 'year_id'])
                dct[(loc_id, sex_id)] = list(cov_sub[['age_group_id', 'year_id', 'mean_value']].values)
    return dct


def intersect(age_start, age_end, year_start, year_end, tuples):
    """
    Find covariate entries that intersects with a given measurement
    entry and compute weights based on length of the overlap.

    Args:
        age_start (int): start age of the measurement entry
        age_end (int): end age of the measurement entry
        year_start (int): start year of the measurement entry
        year_end (int): end year of the entry
        tuples (tuple(int, int, float)): tuples of (age_group_id, year_id, cov_value)
    Returns:
        common_tuples (list[tuple(int, int, float)]): tuples that intersect with measurement
        weights (list[float]): weights corresponding to each tuple
    """
    common_tuples = []
    weights = []
    for tup in tuples:
        tup = (int(tup[0]), int(tup[1]), tup[2])
        age_group = tup[0]
        year = tup[1]
        interval = age_id_to_range[age_group]
        if year_start <= year <= year_end:  # check if intersects in time
            if interval[0] < age_end and interval[1] > age_start:  # check if intersect in age
                common_tuples.append(tup)
                #  pad age_end from data with +1 to account for demographic interval
                weights.append(max(min(age_end+1, interval[1]) - max(age_start, interval[0]), 0) /
                               (interval[1] - interval[0]))
            elif age_start == age_end and \
                    (interval[0] == age_start or interval[1] == age_end):  # case when measurement is on boundary
                common_tuples.append(tup)
                weights.append(1./(interval[1] - interval[0]))
    return common_tuples, weights


def pop_val_dict(df, locations):
    """
    Build a dictionary mapping (location_id, sex_id, age_group_id, year_id) to
    a population value.
    Args:
        df (pandas.Dataframe): population data
        locations (list[int]): location ids

    Returns:
        dct (dict[tuple(int, int, int, int), float])
    """
    dct = {}
    for i, row in df[df['location_id'].isin(locations)].iterrows():
        dct[(row['location_id'], row['sex_id'], row['age_group_id'], row['year_id'])] = row['population']
    return dct


def interpolate(meas, covs, pop):
    """
    Interpolate covariate values for measurements.

    Args:
        meas (pandas.Dataframe): measurement dataframe
        covs (pandas.Dataframe): covariate dataframe
        pop (pandas.Dataframe): population dataframe

    Returns:
        meas (pandas.Dataframe): modified measurement dataframe

    """
    meas = meas.reset_index(drop=True)
    loc_ids = sorted(meas.location_id.unique())
    sex_ids = [1, 2, 3]
    cov_age_year_value = {}
    cov_names = covs.keys()
    t0 = time.time()
    for name, cov in covs.items():
        cov_age_year_value[name] = get_age_year_value(cov, loc_ids, sex_ids)
        meas[name] = 0.0
    print("time elapsed get_age_year_value", time.time() - t0)

    t0 = time.time()
    sex_to_id = {'Male': 1, 'Female': 2, 'Both': 3}
    pop_dict = pop_val_dict(pop, loc_ids)
    print("time elapsed getting pop value", time.time() - t0)

    for i, row in meas.iterrows():
        if (i+1) % 500 == 0:
            print('covariate matching: processed', i+1, 'rows', end='\r')
        loc_id = row['location_id']
        sex_id = sex_to_id[row['sex']]
        age_start = row['age_start']
        age_end = row['age_end']
        year_start = row['year_start']
        year_end = row['year_end']
        dct = {}
        for name in cov_names:
            tuples, weights = intersect(age_start, age_end, year_start, year_end,
                                       cov_age_year_value[name][(loc_id, sex_id)])
            # store only for debugging purpose
            # dct['age_year_'+name] = [(age_id_to_range[tup[0]], tup[1]) for tup in tuples]
            # dct['val_'+name] = [tup[2] for tup in tuples]  # list of covariate values
            # dct['wts'] = weights
            # dct['pop_'+name] = []  # to store list of population values corresponding to tuples
            val = 0.0
            total_wts = 0.0
            if len(tuples) > 0:
                for j in range(len(tuples)):
                    if sex_id != 3:
                        pop_val = pop_dict[(loc_id, sex_id, tuples[j][0], tuples[j][1])]
                        # store only for debugging purpose
                        # dct['pop_'+name].append(pop_dict[(loc_id, sex_id, tuples[j][0], tuples[j][1])])
                    else:
                        pop_val = pop_dict[(loc_id, 1, tuples[j][0], tuples[j][1])] \
                                                + pop_dict[(loc_id, 2, tuples[j][0], tuples[j][1])]
                        # store only for debugging purpose
                        # dct['pop_'+name].append(pop_dict[(loc_id, 1, tuples[j][0], tuples[j][1])]
                        #                        + pop_dict[(loc_id, 2, tuples[j][0], tuples[j][1])])
                    val += tuples[j][2]*weights[j]*pop_val
                    total_wts += weights[j]*pop_val
                    # val += tuples[j][2]*weights[j]*dct['pop_'+name][-1]  # weigh covariate value by population
                    # total_wts += weights[j]*dct['pop_'+name][-1]
                val /= total_wts
            meas.loc[i, name] = val
    return meas
