import itertools
import pandas as pd
import kb_suggestions as sugg

NUM = 3
CAT = 4

categorical_variables = ['Gender', 'Graduat', 'Profession', 'SpendingScore']
numerical_variables = ['Age', 'Experience', 'FamilySize']
selected_features = ['Gender', 'Age', 'Graduat', 'Profession', 'Experience', 'SpendingScore', 'FamilySize']

imputation_techniques_num = ['impute_mean','impute_median','impute_linear_regression','impute_cmeans']
imputation_techniques_cat = ['impute_mode','impute_logistic_regression']
imp_tech_all = ['impute_standard','impute_mode','impute_mean','impute_median','impute_random','impute_knn','impute_mice','impute_linear_regression','impute_logistic_regression','impute_random_forest','impute_cmeans']
od_techniques = ['IQR', 'ISO', 'PERC', 'STD', 'ZSB', 'KNN', 'LOF']

dimensions = ['accuracy','completeness']
algorithms = ['DecisionTree', 'LogisticRegression', 'KNN', 'RandomForest', 'AdaBoost', 'SVC']
ranking = {'DecisionTree': 'completeness', 'LogisticRegression': 'accuracy', 'KNN': 'accuracy', 'RandomForest': 'accuracy', 'AdaBoost': 'completeness', 'SVC': 'accuracy'}
schedule_columns = ['dimension_1', 'dimension_2',  'imp_1', 'imp_2', 'imp_3','imp_4', 'imp_5', 'imp_6','imp_7',
                               'od_1', 'od_2', 'od_3', 'od_4', 'od_5', 'od_6', 'od_7',
                               'imp_col_1', 'imp_col_2', 'imp_col_3', 'imp_col_4', 'imp_col_5', 'imp_col_6', 'imp_col_7', 'algorithm']

def sample_schedule():
    dim = pd.DataFrame(list(itertools.permutations(dimensions)))
    order = list(itertools.permutations(selected_features))
    imp = list(itertools.combinations(imp_tech_all, 7))
    ods = list(itertools.combinations(od_techniques, 7))
    # %%
    order = pd.DataFrame(order)
    imp = pd.DataFrame(imp)
    schedule_compl = imp.merge(order, how='cross')
    schedule_tot = pd.DataFrame(ods).merge(schedule_compl, how='cross')
    schedule_tot.columns = ['0_z', '1_z', '2_z', '3_z', '4_z', '5_z', '6_z', '0_x', '1_x', '2_x',
                            '3_x', '4_x', '5_x', '6_x', '0_y', '1_y', '2_y', '3_y', '4_y', '5_y',
                            '6_y']
    schedule_tot_dim = dim.merge(schedule_tot, how='cross')

    schedule_tot_dim.columns = ['0_k', '1_k', '0_z', '1_z', '2_z', '3_z', '4_z', '5_z', '6_z', '0_x', '1_x', '2_x',
                                '3_x', '4_x', '5_x', '6_x', '0_y', '1_y', '2_y', '3_y', '4_y', '5_y',
                                '6_y']

    schedule_tot_dim.columns = ['dimension_1', 'dimension_2', 'od_1', 'od_2', 'od_3', 'od_4', 'od_5', 'od_6', 'od_7',
                                'imp_1', 'imp_2', 'imp_3', 'imp_4', 'imp_5', 'imp_6', 'imp_7',
                                'imp_col_1', 'imp_col_2', 'imp_col_3', 'imp_col_4', 'imp_col_5', 'imp_col_6',
                                'imp_col_7']

    sample = schedule_tot_dim[
        ['dimension_1', 'dimension_2', 'imp_1', 'imp_2', 'imp_3', 'imp_4', 'imp_5', 'imp_6', 'imp_7',
         'od_1', 'od_2', 'od_3', 'od_4', 'od_5', 'od_6', 'od_7',
         'imp_col_1', 'imp_col_2', 'imp_col_3', 'imp_col_4', 'imp_col_5', 'imp_col_6', 'imp_col_7']]

    sample[['DecisionTree_dirty', 'LogisticRegression_dirty', 'KNN_dirty', 'RandomForest_dirty', 'AdaBoost_dirty', 'SVC_dirty', 'DecisionTree_1', 'LogisticRegression_1', 'KNN_1', 'RandomForest_1', 'AdaBoost_1', 'SVC_1', 'DecisionTree_2',
         'LogisticRegression_2', 'KNN_2', 'RandomForest_2', 'AdaBoost_2', 'SVC_2']] = 0

    ### esclude imputation for numerical/categorical only

    print('Comparing df1...')
    d = sample[((sample.imp_col_1.isin(numerical_variables))) & (sample.imp_1.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df2...')
    d = sample[((sample.imp_col_2.isin(numerical_variables))) & (sample.imp_2.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df3...')
    d = sample[((sample.imp_col_3.isin(numerical_variables))) & (sample.imp_3.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df4...')
    d = sample[((sample.imp_col_4.isin(numerical_variables))) & (sample.imp_4.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df5...')
    d = sample[((sample.imp_col_5.isin(numerical_variables))) & (sample.imp_5.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df6...')
    d = sample[((sample.imp_col_6.isin(numerical_variables))) & (sample.imp_6.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df7...')
    d = sample[((sample.imp_col_7.isin(numerical_variables))) & (sample.imp_7.isin(imputation_techniques_cat))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df8...')
    d = sample[((sample.imp_col_1.isin(categorical_variables))) & (sample.imp_1.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df9...')
    d = sample[((sample.imp_col_2.isin(categorical_variables))) & (sample.imp_2.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df10...')
    d = sample[((sample.imp_col_3.isin(categorical_variables))) & (sample.imp_3.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df11...')
    d = sample[((sample.imp_col_4.isin(categorical_variables))) & (sample.imp_4.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df12...')
    d = sample[((sample.imp_col_5.isin(categorical_variables))) & (sample.imp_5.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df13...')
    d = sample[((sample.imp_col_6.isin(categorical_variables))) & (sample.imp_6.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    print('Comparing df14...')
    d = sample[((sample.imp_col_7.isin(categorical_variables))) & (sample.imp_7.isin(imputation_techniques_num))]
    sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    sample.to_csv('schedule/sample_schedule.csv', index=False)
    return sample, 'schedule/sample_schedule.csv'

def suggested_schedule(df, perc_q):

    ranking = sugg.extract_suggestion_ranking(df, selected_features, perc_q)
    c_tech = sugg.extract_suggestion_completeness(df, selected_features)
    a_tech = sugg.extract_suggestion_accuracy(df, selected_features)
    dim = pd.DataFrame(list(itertools.permutations(dimensions)))

    order = pd.DataFrame(list(itertools.permutations(selected_features)))

    order_algoritm_imputation = order.merge(pd.DataFrame(algorithms), how='cross')
    order_algoritm_imputation.columns = ['imp_col_1', 'imp_col_2', 'imp_col_3','imp_col_4', 'imp_col_5', 'imp_col_6','imp_col_7', 'algorithm']
    order_algoritm_imputation[['imp_1', 'imp_2', 'imp_3','imp_4', 'imp_5', 'imp_6','imp_7']] = str(None)

    for index, row in order_algoritm_imputation.iterrows():
        order_algoritm_imputation.at[index, 'imp_1'] = c_tech[row['algorithm']][row['imp_col_1']]
        order_algoritm_imputation.at[index, 'imp_2'] = c_tech[row['algorithm']][row['imp_col_2']]
        order_algoritm_imputation.at[index, 'imp_3'] = c_tech[row['algorithm']][row['imp_col_3']]

        order_algoritm_imputation.at[index, 'imp_4'] = c_tech[row['algorithm']][row['imp_col_4']]
        order_algoritm_imputation.at[index, 'imp_5'] = c_tech[row['algorithm']][row['imp_col_5']]
        order_algoritm_imputation.at[index, 'imp_6'] = c_tech[row['algorithm']][row['imp_col_6']]
        order_algoritm_imputation.at[index, 'imp_7'] = c_tech[row['algorithm']][row['imp_col_7']]

    order_algoritm_imputation_and_out = order_algoritm_imputation.copy()
    order_algoritm_imputation_and_out[['od_1', 'od_2', 'od_3', 'od_4', 'od_5', 'od_6', 'od_7']] = str(None)

    for index, row in order_algoritm_imputation_and_out.iterrows():
        order_algoritm_imputation_and_out.at[index, 'od_1'] = a_tech[selected_features[0]]
        order_algoritm_imputation_and_out.at[index, 'od_2'] = a_tech[selected_features[1]]
        order_algoritm_imputation_and_out.at[index, 'od_3'] = a_tech[selected_features[2]]

        order_algoritm_imputation_and_out.at[index, 'od_4'] = a_tech[selected_features[3]]
        order_algoritm_imputation_and_out.at[index, 'od_5'] = a_tech[selected_features[4]]
        order_algoritm_imputation_and_out.at[index, 'od_6'] = a_tech[selected_features[5]]
        order_algoritm_imputation_and_out.at[index, 'od_7'] = a_tech[selected_features[6]]

    final_schedule = order_algoritm_imputation_and_out #.merge(order_algoritm_imputation, how='cross')
    #final_schedule = final_schedule[final_schedule.algorithm_x == final_schedule.algorithm_y]
    #final_schedule = final_schedule.drop(columns=['algorithm_y'])

    final_schedule = final_schedule.merge(dim, how='cross')
    #final_schedule = final_schedule.merge(quality, how='cross')

    final_schedule['keep'] = False

    for index, row in final_schedule.iterrows():
        final_schedule.at[index, 'keep'] = True if (row[0] == ranking[row['algorithm']]) else False

    final_schedule = final_schedule[final_schedule.keep == True]

    final_schedule.columns = ['imp_col_1', 'imp_col_2','imp_col_3', 'imp_col_4', 'imp_col_5','imp_col_6','imp_col_7',
                              'algorithm', 'imp_1', 'imp_2', 'imp_3','imp_4', 'imp_5', 'imp_6','imp_7',
                              'od_1', 'od_2', 'od_3', 'od_4', 'od_5', 'od_6', 'od_7', 'dimension_1', 'dimension_2', 'keep']

    final_schedule = final_schedule.reset_index(drop=True)
    final_schedule = final_schedule[schedule_columns]
    final_schedule[['perf_dirty','perf_1', 'perf_2']] = 0
    final_schedule.to_csv('schedule/suggested_schedule.csv', index=False)
    return final_schedule, 'schedule/suggested_schedule.csv'

if __name__ == '__main__':

    #perc_nan = 0.1
    #perc_out = 10
    #perc_q = 0.9

    sample = pd.read_csv("schedule/sample_schedule.csv")


    #print(suggested_schedule(df[selected_features], perc_q, perc_nan, perc_out))
    #sample, name = sample_schedule()

    #schedule = pd.read_csv('schedule/sample_schedule.csv')
    print(len(sample))
