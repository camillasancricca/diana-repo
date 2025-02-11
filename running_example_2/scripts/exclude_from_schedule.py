if __name__ == '__main__':

    import pandas as pd

    categorical_variables = ['Gender', 'Graduat', 'Profession', 'SpendingScore']
    numerical_variables = ['Age', 'Experience', 'FamilySize']
    selected_features = ['Gender', 'Age', 'Graduat', 'Profession', 'Experience', 'SpendingScore', 'FamilySize']

    imputation_techniques_num = ['impute_mean', 'impute_median', 'impute_linear_regression', 'impute_cmeans']
    imputation_techniques_cat = ['impute_mode', 'impute_logistic_regression']
    
    sample = pd.read_csv('scripts/schedule/sample_schedule.csv')

    print(len(sample))
    
    print('Comparing df1...')
    d = sample[((sample.imp_col_1.isin(numerical_variables))) & (sample.imp_1.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df2...')
    d = sample[((sample.imp_col_2.isin(numerical_variables))) & (sample.imp_2.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))

    print('Comparing df3...')
    d = sample[((sample.imp_col_3.isin(numerical_variables))) & (sample.imp_3.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))

    print('Comparing df4...')
    d = sample[((sample.imp_col_4.isin(numerical_variables))) & (sample.imp_4.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df5...')
    d = sample[((sample.imp_col_5.isin(numerical_variables))) & (sample.imp_5.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df6...')
    d = sample[((sample.imp_col_6.isin(numerical_variables))) & (sample.imp_6.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df7...')
    d = sample[((sample.imp_col_7.isin(numerical_variables))) & (sample.imp_7.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df8...')
    d = sample[((sample.imp_col_1.isin(categorical_variables))) & (sample.imp_1.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df9...')
    d = sample[((sample.imp_col_2.isin(categorical_variables))) & (sample.imp_2.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df10...')
    d = sample[((sample.imp_col_3.isin(categorical_variables))) & (sample.imp_3.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df11...')
    d = sample[((sample.imp_col_4.isin(categorical_variables))) & (sample.imp_4.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df12...')
    d = sample[((sample.imp_col_5.isin(categorical_variables))) & (sample.imp_5.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df13...')
    d = sample[((sample.imp_col_6.isin(categorical_variables))) & (sample.imp_6.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))
    
    print('Comparing df14...')
    d = sample[((sample.imp_col_7.isin(categorical_variables))) & (sample.imp_7.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print(len(sample))

    sample.to_csv('scripts/schedule/sample_schedule.csv', index=False)
    
