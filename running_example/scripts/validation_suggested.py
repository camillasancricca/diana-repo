import pandas as pd
import numpy as np
import imputation as i
import outlier_detection as od
import algorithms_class as a
import schedule_validation as sv

df = pd.read_csv("../dataset/mobile.csv")
name_class = 'Price_Class'
selected_features = ['Ratings', 'RAM', 'ROM', 'Mobile_Size',
       'Primary_Cam', 'Selfi_Cam', 'Battery_Power', name_class]
selected_features_only = ['Ratings', 'RAM', 'ROM', 'Mobile_Size',
       'Primary_Cam', 'Selfi_Cam', 'Battery_Power']
df = df[selected_features]

param = {
    'DecisionTree': 100,
    'LogisticRegression': 1,
    'KNN': 3,
    'RandomForest': 250,
    'AdaBoost': 150,
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

def improve_accuracy(df, od_1, od_2, od_3, od_4, od_5, od_6, od_7, imp_1, imp_2, imp_3, imp_4, imp_5, imp_6, imp_7, imp_col_1, imp_col_2, imp_col_3, imp_col_4, imp_col_5, imp_col_6, imp_col_7, algorithm, name_class):
    df_clean = df[selected_features_only].copy()

    indexes_1 = od.outliers(df_clean, od_1, selected_features[0])
    indexes_2 = od.outliers(df_clean, od_2, selected_features[1])
    indexes_3 = od.outliers(df_clean, od_3, selected_features[2])
    indexes_4 = od.outliers(df_clean, od_4, selected_features[3])
    indexes_5 = od.outliers(df_clean, od_5, selected_features[4])
    indexes_6 = od.outliers(df_clean, od_6, selected_features[5])
    indexes_7 = od.outliers(df_clean, od_7, selected_features[6])


    df_clean.loc[indexes_1,selected_features[0]] = np.nan
    df_clean.loc[indexes_2,selected_features[1]] = np.nan
    df_clean.loc[indexes_3,selected_features[2]] = np.nan
    df_clean.loc[indexes_4,selected_features[3]] = np.nan
    df_clean.loc[indexes_5,selected_features[4]] = np.nan
    df_clean.loc[indexes_6,selected_features[5]] = np.nan
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
        df_clean, df_performance_1, df_performance_2 = improve_accuracy(df_clean, sample.od_1, sample.od_2, sample.od_3, sample.od_4, sample.od_5, sample.od_6, sample.od_7, sample.imp_1, sample.imp_2, sample.imp_3, sample.imp_4, sample.imp_5, sample.imp_6, sample.imp_7, sample.imp_col_1, sample.imp_col_2, sample.imp_col_3, sample.imp_col_4, sample.imp_col_5, sample.imp_col_6, sample.imp_col_7, algorithm, name_class)
    else:
        df_clean, df_performance, df_performance_2 = improve_accuracy(df_clean, sample.od_1, sample.od_2, sample.od_3, sample.od_4, sample.od_5, sample.od_6, sample.od_7, sample.imp_1, sample.imp_2, sample.imp_3, sample.imp_4, sample.imp_5, sample.imp_6, sample.imp_7, sample.imp_col_1, sample.imp_col_2, sample.imp_col_3, sample.imp_col_4, sample.imp_col_5, sample.imp_col_6, sample.imp_col_7, algorithm, name_class)

    return df_performance_dirty, df_performance, df_performance_2

def compute_quality(df):

    accuracy = 0

    for col in selected_features_only:
        accuracy += len(od.ZSB(df, col))
    accuracy = 100-(accuracy/df.shape[0]*df.shape[1])*100

    completeness = (df.notnull().sum().sum()/df.shape[0]*df.shape[1])*100

    return round(accuracy+completeness,2)

if __name__ == '__main__':

    all_suggested_schedules = pd.DataFrame([])

    suggested_schedule, file = sv.suggested_schedule(df[selected_features_only], compute_quality(df))
    for index, row in suggested_schedule.iterrows():
        print(" --- Schedule suggested line " + str(index) + " ---")
        perf_dirty, perf_1, perf_2 = validate_sample(df, suggested_schedule.loc[index], row.algorithm, name_class)
        suggested_schedule.loc[index, 'perf_dirty'] = perf_dirty[row.algorithm]
        suggested_schedule.loc[index, 'perf_1'] = perf_1[row.algorithm]
        suggested_schedule.loc[index, 'perf_2'] = perf_2[row.algorithm]
    all_suggested_schedules = pd.concat([all_suggested_schedules, suggested_schedule], ignore_index=True)

    all_suggested_schedules.to_csv('schedule/compiled_schedules.csv', index=False)
