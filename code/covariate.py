import numpy as np
import time
import warnings
warnings.simplefilter('error')

age_intervals = [(0, 7./365), (7./365, 28./365), (28./365, 1)]+[(0, 1), (1, 5)] + \
                [(x, x+5) for x in np.arange(5, 100, 5)] + [(0, 100)]
age_group_ids = [2, 3, 4] + [28, 5] + list(range(6, 21)) + list(range(30, 34)) + [22]
assert len(age_intervals) == len(age_group_ids)
n_age = len(age_group_ids)
age_id_to_range = {age_group_ids[i]: age_intervals[i] for i in range(n_age)}


def get_age_year_value(cov, loc_ids, sex_ids):
    dct = {}
    avail_locs = set(cov.location_id.unique())
    for loc_id in loc_ids:
        for sex_id in sex_ids:
            if loc_id in avail_locs:
                cov_sub = cov[(cov['location_id'] == loc_id) & (cov['sex_id'].isin([3, sex_id]))]
                cov_sub = cov_sub.sort_values(['age_group_id', 'year_id'])
                dct[(loc_id, sex_id)] = list(cov_sub[['age_group_id', 'year_id', 'mean_value']].values)
            else:
                dct[(loc_id, sex_id)] = []
    return dct


def intersect(age_start, age_end, year_start, year_end, tuples):
    common_tuples = []
    weights = []
    for tup in tuples:
        tup = (int(tup[0]), int(tup[1]), tup[2])
        age_group = tup[0]
        if age_group in age_id_to_range:
            year = tup[1]
            interval = age_id_to_range[age_group]
            if year >= year_start and year <= year_end:
                if interval[0] < age_end and interval[1] > age_start:
                    common_tuples.append(tup)
                    weights.append(max(min(age_end+1, interval[1]) - max(age_start, interval[0]), 0) /
                                   (interval[1] - interval[0]))
                elif age_start == age_end and \
                     (interval[0] == age_start or interval[1] == age_end):
                    common_tuples.append(tup)
                    weights.append(1./(interval[1] - interval[0]))
    return common_tuples, weights


def pop_val_dict(df, locations):
    dct = {}
    for i, row in df[df['location_id'].isin(locations)].iterrows():
        dct[(row['location_id'], row['sex_id'], row['age_group_id'], row['year_id'])] = row['population']
    return dct


def interpolate(meas, covs, pop):
    meas = meas.reset_index(drop=True)
    loc_ids = sorted(meas.location_id.unique())
    sex_ids = [1, 2, 3]
    cov_age_year_value = {}
    cov_names = covs.keys()
    for name, cov in covs.items():
        cov_age_year_value[name] = get_age_year_value(cov, loc_ids, sex_ids)
        meas[name] = 0.0

    sex_to_id = {'Male': 1, 'Female': 2, 'Both': 3}
    pop_dict = pop_val_dict(pop, loc_ids)

    for i, row in meas.iterrows():
        if (i+1) % 500 == 0:
            print('processed', i+1, 'rows', end='\r')
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
            dct['age_year_'+name] = [(age_id_to_range[tup[0]], tup[1]) for tup in tuples]
            dct['val_'+name] = [tup[2] for tup in tuples]
            dct['wts'] = weights
            dct['pop_'+name] = []
            val = 0.0
            total_wts = 0.0
            if len(tuples) > 0:
                for j in range(len(tuples)):
                    # t1 = time.time()
                    #                     # pop_value = pop[(pop['location_id'] == loc_id)\
                    #                     #                & (pop['age_group_id'] == tuples[j][0]) & \
                    #                     #                (pop['year_id'] == tuples[j][1])][['population', 'sex_id']]
                    #                     # if sex_id != 3:
                    #                     #     dct['pop_'+name].append(pop_value[pop_value['sex_id'] == sex_id]['population'].values)
                    #                     # else:
                    #                     #     dct['pop_'+name].append(np.sum(pop_value['population'].values))
                    #                     # print(i, name, j, 'get pop val elapsed', time.time() - t1)
                    if sex_id != 3:
                        dct['pop_'+name].append(pop_dict[(loc_id, sex_id, tuples[j][0], tuples[j][1])])
                    else:
                        dct['pop_'+name].append(pop_dict[(loc_id, 1, tuples[j][0], tuples[j][1])]
                                                + pop_dict[(loc_id, 2, tuples[j][0], tuples[j][1])])
                    val += tuples[j][2]*weights[j]*dct['pop_'+name][-1]
                    total_wts += weights[j]*dct['pop_'+name][-1]
                val /= total_wts
            meas.loc[i, name] = val
