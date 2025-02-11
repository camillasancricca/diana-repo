import pandas as pd

kb_completeness = pd.read_csv("../kb/KBC.csv")

techniques = ['impute_standard','impute_mean','impute_median','impute_random','impute_knn','impute_mice','impute_linear_regression','impute_random_forest','impute_cmeans']

cols = ['name', 'column_name', 'n_tuples', 'missing_perc', 'uniqueness', 'min',
       'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr',
       'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density', 'ml_algorithm', 'impute_standard_impact', 'impute_mean_impact', 'impute_median_impact',
       'impute_random_impact', 'impute_knn_impact', 'impute_mice_impact',
       'impute_linear_regression_impact', 'impute_random_forest_impact',
       'impute_cmeans_impact']

def get_kb_completeness():

    kb_ = kb_completeness.drop_duplicates()

    return kb_

### TODO AFTER DIFF
def get_kb_impact_completeness():

    kb_ = get_kb_completeness()

    kb_new = kb_.copy()

    kb_new = kb_new.drop_duplicates()

    ### impact = 1-df_clean/df_standard_value

    for tech in techniques:
        kb_new[tech + '_impact'] = 1 - kb_new[tech] / kb_new['impute_standard']

    kb_new = kb_new[cols]

    for tech in techniques:
        kb_new = kb_new.rename(columns={tech+'_impact': tech})

    return kb_new
