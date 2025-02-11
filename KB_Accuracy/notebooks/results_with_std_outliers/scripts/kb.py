import pandas as pd
import numpy as np

data_profile = pd.read_csv("../kb/KBA_profiling.csv")
data_results = pd.read_csv("../kb/KBA_results.csv")

techniques = data_results.technique_accuracy.unique()
stds_outliers = data_results.std_outliers.unique()
perc_outliers = data_results.percentage_outliers.unique()
objects = data_results.column.unique()
datasets = data_profile.name.unique()

data_results["percentage_outliers"] = data_results["percentage_outliers"].astype(float)
data_profile["std_outliers"] = data_profile["std_outliers"].astype(float)

columns_results = ['percentage_outliers','std_outliers', 'technique_accuracy', 'f1_technique']

columns_profile = ['name', 'column_name', 'n_tuples', 'uniqueness',
       'min', 'max', 'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad',
       'iqr', 'p_min', 'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy',
       'density']

def get_kb_accuracy():
    new_kb_accuracy = pd.DataFrame([])

    for dataset in datasets:
        objects = data_results[data_results["name"] == dataset].column.unique()

        for object in objects:

            for perc in perc_outliers:
                for std in stds_outliers:

                    df_1 = data_results[(data_results["name"] == dataset) & (data_results["column"] == object) & (data_results["percentage_outliers"] == perc) & (data_results["std_outliers"] == std)]
                    df_1 = df_1[columns_results].reset_index(drop=True).copy()

                    df_new = data_profile[(data_profile["name"] == dataset) & (data_profile["column_name"] == object) & (data_profile["percentage_outliers"] == perc) & (data_profile["std_outliers"] == std)]
                    newdf = pd.DataFrame(np.repeat(df_new.values, len(df_1), axis=0),columns=df_new.columns)
                    df_2 = newdf[columns_profile].reset_index(drop=True).copy()

                    df_3 = pd.concat([df_2, df_1], ignore_index=False, axis=1)

                    new_kb_accuracy = pd.concat([new_kb_accuracy, df_3], ignore_index=True)

    return new_kb_accuracy

if __name__ == '__main__':
    kb = get_kb_accuracy()
    kb.to_csv("KBA.csv",index=None)
