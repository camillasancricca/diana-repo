import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import root_mean_squared_error
#from .kb import get_kb_accuracy

kb_accuracy = pd.read_csv("KBA.csv")

techniques = kb_accuracy.technique_accuracy.unique()
perc_outliers = kb_accuracy.percentage_outliers.unique()
objects = kb_accuracy.column_name.unique()
datasets = kb_accuracy.name.unique()

columns = ['name', 'column_name', 'technique_accuracy']

columns_X = ['n_tuples', 'uniqueness', 'min', 'max',
       'mean', 'median', 'std', 'skewness', 'kurtosis', 'mad', 'iqr', 'p_min',
       'p_max', 'k_min', 'k_max', 's_min', 's_max', 'entropy', 'density',
       'percentage_outliers']

columns_y = 'f1_technique'

def training_testing_accuracy():
    with open("../../results/prediction_accuracy.csv", "w") as f1:
        f1.write("dataset,technique,rmse\n")

        with open("../../results/techniques_accuracy_evaluation.csv", "w") as f2:
            f2.write("dataset,technique,real,pred,perc_outliers\n")

            for dataset in datasets:
                    for technique in techniques:

                        data = kb_accuracy.copy()

                        df = data[(data["technique_accuracy"] == technique)].copy()

                        train = df[df["name"] != dataset]
                        test = df[df["name"] == dataset]

                        X_train = train[columns_X]
                        y_train = train[columns_y]
                        X_test = test[columns_X]
                        y_test = test[columns_y]

                        X_test_not_scaled = X_test.reset_index(drop=True).copy()

                        X_train = StandardScaler().fit_transform(X_train)
                        X_train = np.nan_to_num(X_train)

                        X_test = StandardScaler().fit_transform(X_test)
                        X_test = np.nan_to_num(X_test)

                        knn = KNeighborsRegressor(n_neighbors=27, metric='manhattan')
                        knn.fit(X_train, y_train)

                        y_pred = knn.predict(X_test)
                        error = root_mean_squared_error(y_test, y_pred)
                        print(dataset+": "+str(error))
                        f1.write(dataset + "," + technique + "," + str(error) + "\n")

                        y_test = y_test.reset_index(drop=True)
                        for i in range(0, len(y_test)):
                            f2.write(dataset + "_" + str(i) + "," + technique + "," + str(
                                y_test[i]) + "," + str(y_pred[i]) + "," + str(X_test_not_scaled.percentage_outliers[i]) + "\n")

    data = pd.read_csv("../../results/prediction_accuracy.csv")
    print("Done! Final RMSE: "+str(data.rmse.mean()))


def evaluate_techniques():

    rankings = pd.read_csv("../../results/techniques_accuracy_evaluation.csv")

    with open("../../results/techniques_accuracy_evaluation_total.csv", "w") as f:
        f.write("dataset,ranking_eval,ranking_real,ranking_pred,value_real,value_pred,perc_outliers\n")
        datasets = rankings.dataset.unique()
        for dataset in datasets:
                rrr = rankings[(rankings["dataset"] == dataset)].copy()

                rrr = rrr.reset_index(drop=True)

                df_real = rrr.sort_values(by=['real'], ascending=False).copy().reset_index(drop=True)
                df_pred = rrr.sort_values(by=['pred'], ascending=False).copy().reset_index(drop=True)

                count = 0
                for i in range(0, len(df_real)):
                    if df_real.loc[i].technique == df_pred.loc[i].technique:
                        count = count + 1

                f.write(dataset + "," + str(count / len(df_real)) + "," + str(
                    np.array(df_real.technique)).replace('\n', '') + "," + str(np.array(df_pred.technique)).replace(
                    '\n', '') + "," + str(
                    np.array(df_real.real)).replace('\n', '') + "," + str(np.array(df_pred.pred)).replace('\n',
                                                                                                          '') + "," + str(rrr.perc_outliers[0]) + "\n")
    print("Done!")

if __name__ == '__main__':
    training_testing_accuracy()
    evaluate_techniques()
