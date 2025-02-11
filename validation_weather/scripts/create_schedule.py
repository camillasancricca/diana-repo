import itertools

import pandas as pd

if __name__ == '__main__':

    selected_features = ['Temperature','Precipitation','AtmosphericPressure']
    imputation_techniques = ['impute_standard','impute_mean','impute_median','impute_random','impute_knn','impute_mice','impute_linear_regression','impute_random_forest','impute_cmeans']
    od_techniques = ['IQR', 'ISO', 'PERC', 'STD', 'ZSB', 'KNN', 'LOF']
    dimensions = ['accuracy','completeness']
    quality = pd.DataFrame([60,65,70,75,80,85,90,95])
    dim = pd.DataFrame(list(itertools.permutations(dimensions)))
    order = list(itertools.permutations(selected_features))
    imp = list(itertools.combinations(imputation_techniques,3))
    ods = list(itertools.combinations(od_techniques,3))
    ###schedule
    order = pd.DataFrame(order)
    imp = pd.DataFrame(imp)
    schedule_compl = imp.merge(order, how='cross')
    print('ok')
    schedule_acc = pd.DataFrame(ods).merge(schedule_compl, how='cross')
    print('ok')
    schedule_tot = schedule_compl.merge(schedule_acc, how='cross')
    print('ok')
    schedule_tot_dim = dim.merge(schedule_tot, how='cross')
    print('ok')
    schedule_tot_perc = schedule_tot_dim.merge(quality, how='cross')
    print('ok')
    schedule_tot_perc.columns = ['dimension_1','dimension_2','imp_1','imp_2','imp_3','imp_col_1','imp_col_2','imp_col_3','od_1','od_2','od_3','od_imp_1','od_imp_2','od_imp_3','od_imp_col_1','od_imp_col_2','od_imp_col_3','quality']
    schedule_tot_perc.to_csv('schedule.csv', index=False)
