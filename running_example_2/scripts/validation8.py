import pandas as pd
import numpy as np
import imputation as i
import outlier_detection as od
import algorithms_class as a
import schedule_validation as sv

df = pd.read_csv("../dataset/people.csv")
name_class = 'Married'
selected_features = ['Gender', 'Age', 'Graduat', 'Profession', 'Experience',
        'SpendingScore', 'FamilySize', name_class]
selected_features_only = ['Gender', 'Age', 'Graduat', 'Profession', 'Experience',
        'SpendingScore', 'FamilySize']
df = df[selected_features]

param = {
    'DecisionTree': 200,
    'LogisticRegression': 1,
    'KNN': 30,
    'RandomForest': 250,
    'AdaBoost': 100,
    'SVC': 1
}

models = ['DecisionTree','LogisticRegression','KNN','RandomForest','AdaBoost','SVC']

def perform_analysis(df, name_class, algorithm):
    performance = {}
    if algorithm != None:
        performance[algorithm] = a.classification(df[selected_features_only], df[name_class], algorithm, param[algorithm], 4)
    else:
        for m in models:
            performance[m] = a.classification(df[selected_features_only], df[name_class], m, param[m], 4)
    return performance

def improve_completeness(df, imp_1, imp_2, imp_3, imp_4, imp_5, imp_6, imp_7, imp_col_1, imp_col_2, imp_col_3, imp_col_4, imp_col_5, imp_col_6, imp_col_7, algorithm, name_class):
    df_clean = df[selected_features_only].copy()

    if df_clean[imp_col_1].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_1, imp_col_1)

    if df_clean[imp_col_2].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_2, imp_col_2)

    if df_clean[imp_col_3].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_3, imp_col_3)

    if df_clean[imp_col_4].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_4, imp_col_4)

    if df_clean[imp_col_5].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_5, imp_col_5)

    if df_clean[imp_col_6].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_6, imp_col_6)


    if df_clean[imp_col_7].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_7, imp_col_7)

    df_clean[name_class] = df[name_class]

    df_performance = perform_analysis(df_clean, name_class, algorithm)

    return df_clean, df_performance

def improve_accuracy(df, od_1, od_2, od_3, imp_1, imp_2, imp_3, imp_4, imp_5, imp_6, imp_7, imp_col_1, imp_col_2, imp_col_3, imp_col_4, imp_col_5, imp_col_6, imp_col_7, algorithm, name_class):
    df_clean = df[selected_features_only].copy()

    indexes_2 = od.outliers(df_clean, od_1, selected_features[1])
    indexes_5 = od.outliers(df_clean, od_2, selected_features[4])
    indexes_7 = od.outliers(df_clean, od_3, selected_features[6])

    df_clean.loc[indexes_2,selected_features[1]] = np.nan
    df_clean.loc[indexes_5,selected_features[4]] = np.nan
    df_clean.loc[indexes_7,selected_features[6]] = np.nan

    df_clean[name_class] = df[name_class]

    df_performance_1 = perform_analysis(df_clean, name_class, algorithm)

    df_clean, df_performance_2 = improve_completeness(df_clean, imp_1, imp_2, imp_3, imp_4, imp_5, imp_6, imp_7, imp_col_1, imp_col_2, imp_col_3, imp_col_4, imp_col_5, imp_col_6, imp_col_7, algorithm, name_class)

    return df_clean, df_performance_1, df_performance_2

def validate_sample(df, sample, algorithm, name_class):
    df_clean = df.copy()

    df_performance_dirty = perform_analysis(df_clean, name_class, algorithm)

    if sample.dimension_1 == 'completeness':
        df_clean, df_performance = improve_completeness(df_clean, sample.imp_1, sample.imp_2, sample.imp_3, sample.imp_4, sample.imp_5, sample.imp_6, sample.imp_7, sample.imp_col_1, sample.imp_col_2, sample.imp_col_3, sample.imp_col_4, sample.imp_col_5, sample.imp_col_6, sample.imp_col_7, algorithm, name_class)
        df_clean, df_performance_1, df_performance_2 = improve_accuracy(df_clean, sample.od_2, sample.od_5, sample.od_7,
                                                                        sample.imp_1, sample.imp_2, sample.imp_3,
                                                                        sample.imp_4, sample.imp_5, sample.imp_6,
                                                                        sample.imp_7, sample.imp_col_1,
                                                                        sample.imp_col_2, sample.imp_col_3,
                                                                        sample.imp_col_4, sample.imp_col_5,
                                                                        sample.imp_col_6, sample.imp_col_7, algorithm,
                                                                        name_class)
    else:
        df_clean, df_performance, df_performance_2 = improve_accuracy(df_clean, sample.od_2, sample.od_5, sample.od_7,
                                                                      sample.imp_1, sample.imp_2, sample.imp_3,
                                                                      sample.imp_4, sample.imp_5, sample.imp_6,
                                                                      sample.imp_7, sample.imp_col_1, sample.imp_col_2,
                                                                      sample.imp_col_3, sample.imp_col_4,
                                                                      sample.imp_col_5, sample.imp_col_6,
                                                                      sample.imp_col_7, algorithm, name_class)

    return df_performance_dirty, df_performance, df_performance_2

if __name__ == '__main__':

    #sample_schedule, file = sv.sample_schedule()
    sample_schedule = pd.read_csv('schedule/sample_schedule.csv')
    sample_schedule = sample_schedule[160000:180000]

    for index, row in sample_schedule.iterrows():
        print(" --- Schedule sample line "+str(index)+" ---")
        perf_dirty, perf_1, perf_2 = validate_sample(df, sample_schedule.loc[index], None, name_class)

        for m in models:
            sample_schedule.loc[index,m+'_dirty'] = perf_dirty[m]
            sample_schedule.loc[index,m+'_1'] = perf_1[m]
            sample_schedule.loc[index,m+'_2'] = perf_2[m]

    sample_schedule.to_csv('schedule/compiled_sample_schedule8.csv', index=False)
