import itertools
import pandas as pd
import kb_suggestions as sugg

NUM = 2
CAT = 1

selected_features = ['Strength','Intelligence','Weaknesses']

numerical_feature1 = 'Strength'
numerical_feature2 = 'Intelligence'
categorical_feature1 = 'Weaknesses'

numerical_features = [numerical_feature1,numerical_feature2]
categorical_features = [categorical_feature1]

imputation_techniques_num = ['impute_mean','impute_median','impute_linear_regression','impute_cmeans']
imputation_techniques_cat = ['impute_mode','impute_logistic_regression']

imp_tech_all = ['impute_standard','impute_mode','impute_mean','impute_median','impute_random','impute_knn','impute_mice','impute_linear_regression','impute_logistic_regression','impute_random_forest','impute_cmeans']

od_techniques = ['IQR', 'ISO', 'PERC', 'STD', 'ZSB', 'KNN', 'LOF']
dimensions = ['accuracy','completeness']
quality = pd.DataFrame([50,60,70,80,90])
algorithms = ['DecisionTree', 'LogisticRegression', 'KNN', 'RandomForest', 'AdaBoost', 'SVC']
ranking = {'DecisionTree': 'completeness', 'LogisticRegression': 'accuracy', 'KNN': 'accuracy', 'RandomForest': 'accuracy', 'AdaBoost': 'completeness', 'SVC': 'accuracy'}
c_tech = {'DecisionTree': {'Temperature': 'impute_standard', 'Precipitation': 'impute_standard', 'AtmosphericPressure': 'impute_standard'}, 'LogisticRegression': {'Temperature': 'impute_random_forest', 'Precipitation': 'impute_random_forest', 'AtmosphericPressure': 'impute_random_forest'}, 'KNN': {'Temperature': 'impute_median', 'Precipitation': 'impute_median', 'AtmosphericPressure': 'impute_median'}, 'RandomForest': {'Temperature': 'impute_standard', 'Precipitation': 'impute_standard', 'AtmosphericPressure': 'impute_standard'}, 'AdaBoost': {'Temperature': 'impute_standard', 'Precipitation': 'impute_standard', 'AtmosphericPressure': 'impute_standard'}, 'SVC': {'Temperature': 'impute_knn', 'Precipitation': 'impute_knn', 'AtmosphericPressure': 'impute_knn'}}
a_tech = {'Temperature': 'IQR', 'Precipitation': 'ZSB', 'AtmosphericPressure': 'ZSB'}
schedule_columns = ['dimension_1', 'dimension_2', 'imp_1', 'imp_2', 'imp_3', 'od_1', 'od_2', 'imp_col_1', 'imp_col_2','imp_col_3', 'quality', 'algorithm']

def sample_schedule():
    dim = pd.DataFrame(list(itertools.permutations(dimensions)))
    order = list(itertools.permutations(selected_features))
    imp = list(itertools.combinations(imp_tech_all, 3))
    ods = list(itertools.combinations(od_techniques, 2))
    # %%
    order = pd.DataFrame(order)
    imp = pd.DataFrame(imp)

    schedule_compl = imp.merge(order, how='cross')

    schedule_tot = pd.DataFrame(ods).merge(schedule_compl, how='cross')
    schedule_tot.columns = ['0_z', '1_z', '0_x', '1_x', '2_x', '0_y', '1_y', '2_y']
    schedule_tot_dim = dim.merge(schedule_tot, how='cross')
    schedule_tot_dim.columns = ['0_k', '1_k', '0_z', '1_z', '0_x', '1_x', '2_x', '0_y', '1_y', '2_y']
    schedule_tot_perc = schedule_tot_dim.merge(quality, how='cross')
    schedule_tot_perc.columns = ['dimension_1', 'dimension_2', 'od_1', 'od_2', 'imp_1', 'imp_2', 'imp_3',
                                 'imp_col_1', 'imp_col_2',
                                 'imp_col_3', 'quality']
    sample = schedule_tot_perc[['dimension_1', 'dimension_2', 'imp_1', 'imp_2', 'imp_3', 'od_1', 'od_2',
                                 'imp_col_1', 'imp_col_2',
                                 'imp_col_3', 'quality']]
    sample[['DecisionTree_dirty', 'LogisticRegression_dirty', 'KNN_dirty', 'RandomForest_dirty', 'AdaBoost_dirty', 'SVC_dirty', 'DecisionTree_1', 'LogisticRegression_1', 'KNN_1', 'RandomForest_1', 'AdaBoost_1', 'SVC_1', 'DecisionTree_2',
         'LogisticRegression_2', 'KNN_2', 'RandomForest_2', 'AdaBoost_2', 'SVC_2']] = 0

    ### esclude imputation for numerical/categorical only

    df1 = sample[((sample.imp_col_1 == numerical_feature1)) & (sample.imp_1.isin(imputation_techniques_cat))]
    df2 = sample[((sample.imp_col_2 == numerical_feature1)) & (sample.imp_2.isin(imputation_techniques_cat))]
    df3 = sample[((sample.imp_col_3 == numerical_feature1)) & (sample.imp_3.isin(imputation_techniques_cat))]

    df4 = sample[((sample.imp_col_1 == numerical_feature2)) & (sample.imp_1.isin(imputation_techniques_cat))]
    df5 = sample[((sample.imp_col_2 == numerical_feature2)) & (sample.imp_2.isin(imputation_techniques_cat))]
    df6 = sample[((sample.imp_col_3 == numerical_feature2)) & (sample.imp_3.isin(imputation_techniques_cat))]

    df7 = sample[((sample.imp_col_1 == categorical_feature1) | (sample.imp_col_1 == categorical_feature1)) & (sample.imp_1.isin(imputation_techniques_num))]
    df8 = sample[((sample.imp_col_2 == categorical_feature1) | (sample.imp_col_2 == categorical_feature1)) & (sample.imp_2.isin(imputation_techniques_num))]
    df9 = sample[((sample.imp_col_3 == categorical_feature1) | (sample.imp_col_3 == categorical_feature1)) & (sample.imp_3.isin(imputation_techniques_num))]

    for d in [df1, df2, df3, df4, df5, df6, df7, df8, df9]:
        sample = sample[~sample.apply(tuple, 1).isin(d.apply(tuple, 1))]

    sample.to_csv('schedule/sample_schedule.csv', index=False)
    return sample, 'schedule/sample_schedule.csv'

def suggested_schedule(df, perc_q, perc_nan, perc_out):

    ranking = sugg.extract_suggestion_ranking(df, selected_features, perc_q)
    c_tech = sugg.extract_suggestion_completeness(df, selected_features, perc_nan)
    a_tech = sugg.extract_suggestion_accuracy(df, selected_features, perc_out)
    dim = pd.DataFrame(list(itertools.permutations(dimensions)))

    order = pd.DataFrame(list(itertools.permutations(selected_features)))

    order_algoritm_imputation = order.merge(pd.DataFrame(algorithms), how='cross')
    order_algoritm_imputation.columns = ['imp_col_1', 'imp_col_2', 'imp_col_3', 'algorithm']
    order_algoritm_imputation[['imp_1', 'imp_2', 'imp_3']] = str(None)

    for index, row in order_algoritm_imputation.iterrows():
        order_algoritm_imputation.at[index, 'imp_1'] = c_tech[row['algorithm']][row['imp_col_1']]
        order_algoritm_imputation.at[index, 'imp_2'] = c_tech[row['algorithm']][row['imp_col_2']]
        order_algoritm_imputation.at[index, 'imp_3'] = c_tech[row['algorithm']][row['imp_col_3']]

    order_algoritm_imputation_and_out = order_algoritm_imputation.copy()
    order_algoritm_imputation_and_out[['od_1']] = str(None)
    order_algoritm_imputation_and_out[['od_2']] = str(None)

    for index, row in order_algoritm_imputation_and_out.iterrows():
        order_algoritm_imputation_and_out.at[index, 'od_1'] = a_tech[numerical_feature1]
        order_algoritm_imputation_and_out.at[index, 'od_2'] = a_tech[numerical_feature2]
        #order_algoritm_imputation_and_out.at[index, 'od_3'] = a_tech[selected_features[2]]

    final_schedule = order_algoritm_imputation_and_out #.merge(order_algoritm_imputation, how='cross')
    #final_schedule = final_schedule[final_schedule.algorithm_x == final_schedule.algorithm_y]
    #final_schedule = final_schedule.drop(columns=['algorithm_y'])

    final_schedule = final_schedule.merge(dim, how='cross')
    #final_schedule = final_schedule.merge(quality, how='cross')

    final_schedule['keep'] = False

    for index, row in final_schedule.iterrows():
        final_schedule.at[index, 'keep'] = True if (row[0] == ranking[row['algorithm']]) else False

    final_schedule = final_schedule[final_schedule.keep == True]

    final_schedule.columns = ['imp_col_1', 'imp_col_2', 'imp_col_3', 'algorithm', 'imp_1', 'imp_2', 'imp_3', 'od_1', 'od_2',
                            'dimension_1', 'dimension_2', 'keep']

    final_schedule = final_schedule.reset_index(drop=True)
    final_schedule[['quality']] = perc_q
    final_schedule = final_schedule[schedule_columns]
    final_schedule[['perf_dirty','perf_1', 'perf_2']] = 0
    final_schedule.to_csv('schedule/suggested_schedule.csv', index=False)
    return final_schedule, 'schedule/suggested_schedule.csv'

if __name__ == '__main__':

    perc_nan = 0.1
    perc_out = 10
    perc_q = 0.9

    df = pd.read_csv("../dataset/character.csv")

    #print(suggested_schedule(df[selected_features], perc_q, perc_nan, perc_out))
    print(sample_schedule())
