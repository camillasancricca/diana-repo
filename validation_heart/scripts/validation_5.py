import pandas as pd
import dirty_data as d
import numpy as np
import imputation as i
import outlier_detection as od
import algorithms_class as a


df = pd.read_csv("../dataset/heart.csv")
name_class = 'disease'

selected_features = ['oldpeak','cp','thal', name_class]
selected_features_only = ['oldpeak','cp','thal']
df = df[selected_features]
quality = pd.DataFrame([50,60,70,80,90])
perc_quality = [50,60,70,80,90]

param = {
    'DecisionTree': 70,
    'LogisticRegression': 1,
    'KNN': 8,
    'RandomForest': 70,
    'AdaBoost': 90,
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

def improve_completeness(df, imp_1, imp_2, imp_3, imp_col_1, imp_col_2, imp_col_3, algorithm, name_class):
    df_clean = df[selected_features_only].copy()

    if df_clean[imp_col_1].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_1, imp_col_1)

    if df_clean[imp_col_2].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_2, imp_col_2)

    if df_clean[imp_col_3].isnull().sum() != 0:
        df_clean = i.impute(df_clean, imp_3, imp_col_3)

    df_clean[name_class] = df[name_class]

    df_performance = perform_analysis(df_clean, name_class, algorithm)

    return df_clean, df_performance

def improve_accuracy(df, od_1, imp_1, imp_2, imp_3, imp_col_1, imp_col_2, imp_col_3, algorithm, name_class):
    df_clean = df[selected_features_only].copy()

    indexes_1 = od.outliers(df_clean, od_1, selected_features[0])

    df_clean.loc[indexes_1,selected_features[0]] = np.nan

    df_clean[name_class] = df[name_class]

    df_performance_1 = perform_analysis(df_clean, name_class, algorithm)

    df_clean, df_performance_2 = improve_completeness(df_clean, imp_1, imp_2, imp_3, imp_col_1, imp_col_2, imp_col_3, algorithm, name_class)

    return df_clean, df_performance_1, df_performance_2

def validate_sample(df, sample, algorithm, name_class):
    df_clean = df.copy()

    df_performance_dirty = perform_analysis(df_clean, name_class, algorithm)

    if sample.dimension_1 == 'completeness':
        df_clean, df_performance = improve_completeness(df_clean, sample.imp_1, sample.imp_2, sample.imp_3, sample.imp_col_1, sample.imp_col_2, sample.imp_col_3, algorithm, name_class)
        df_clean, df_performance_1, df_performance_2 = improve_accuracy(df_clean, sample.od_1, sample.imp_1, sample.imp_2, sample.imp_3, sample.imp_col_1, sample.imp_col_2, sample.imp_col_3, algorithm, name_class)
    else:
        df_clean, df_performance, df_performance_2 = improve_accuracy(df_clean, sample.od_1, sample.imp_1, sample.imp_2, sample.imp_3, sample.imp_col_1, sample.imp_col_2, sample.imp_col_3, algorithm, name_class)

    return df_performance_dirty, df_performance, df_performance_2

if __name__ == '__main__':

    sample_schedule = pd.read_csv('schedule/sample_schedule.csv')
    sample_schedule = sample_schedule[6400:8000]

    for index, row in sample_schedule.iterrows():
        print(" --- Schedule sample line "+str(index)+" ---")
        perc = row.quality/100
        df_dirt = d.injection(df, name_class, perc, 10, 1)
        perf_dirty, perf_1, perf_2 = validate_sample(df_dirt, sample_schedule.loc[index], None, name_class)

        for m in models:
            sample_schedule.loc[index,m+'_dirty'] = perf_dirty[m]
            sample_schedule.loc[index,m+'_1'] = perf_1[m]
            sample_schedule.loc[index,m+'_2'] = perf_2[m]

    sample_schedule.to_csv('schedule/compiled_sample_schedule_5.csv', index=False)
