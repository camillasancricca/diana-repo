import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import root_mean_squared_error
from .kb import get_kb_impact_completeness

kb_completeness = get_kb_impact_completeness()

datasets = kb_completeness.name.unique()
objects = kb_completeness.column_name.unique()
ml_algorithms = kb_completeness.ml_algorithm.unique()

columns_X = ['n_tuples', 'constancy',
       'imbalance', 'uniqueness', 'unalikeability', 'entropy', 'density',
       'mean_char', 'std_char', 'skewness_char', 'kurtosis_char', 'min_char',
       'max_char', 'missing_perc']

techniques = ['impute_standard', 'impute_mode',
       'impute_random', 'impute_knn', 'impute_mice',
       'impute_logistic_regression', 'impute_random_forest', 'impute_kproto']

def training_testing_completeness():
    with open("../results/prediction_completeness.csv", "w") as f1:
        f1.write("dataset,model,technique,rmse\n")

        with open("../results/techniques_completeness_evaluation.csv", "w") as f2:
            f2.write("dataset,model,technique,real,pred,perc_completeness\n")

            for dataset in datasets:
                for model in ml_algorithms:
                    for technique in techniques:

                        data = kb_completeness.copy()

                        df = data[(data["ml_algorithm"] == model)].copy()

                        train = df[df["name"] != dataset]
                        test = df[df["name"] == dataset]

                        X_train = train[columns_X]
                        y_train = train[technique]
                        X_test = test[columns_X]
                        y_test = test[technique]

                        X_test_not_scaled = X_test.reset_index(drop=True).copy()

                        X_train = StandardScaler().fit_transform(X_train)
                        X_train = np.nan_to_num(X_train)

                        X_test = StandardScaler().fit_transform(X_test)
                        X_test = np.nan_to_num(X_test)

                        knn = KNeighborsRegressor(n_neighbors=35, metric='manhattan')
                        knn.fit(X_train, y_train)

                        y_pred = knn.predict(X_test)
                        error = root_mean_squared_error(y_test, y_pred)
                        print(dataset+": "+str(error))
                        f1.write(dataset + "," + model + "," + technique + "," + str(error) + "\n")

                        y_test = y_test.reset_index(drop=True)
                        for i in range(0, len(y_test)):
                            f2.write(dataset + "_" + str(i) + "," + model + "," + technique + "," + str(
                                y_test[i]) + "," + str(y_pred[i]) + "," + str(X_test_not_scaled.missing_perc[i]) + "\n")

    data = pd.read_csv("../results/prediction_completeness.csv")
    print("Done! Final RMSE: "+str(data.rmse.mean()))




def evaluate_techniques():

    rankings = pd.read_csv("../results/techniques_completeness_evaluation.csv")

    with open("../results/techniques_completeness_evaluation_total.csv", "w") as f:
        f.write("dataset,model,ranking_eval,ranking_real,ranking_pred,value_real,value_pred,perc_completeness\n")
        datasets = rankings.dataset.unique()
        for dataset in datasets:
            for model in ml_algorithms:
                rrr = rankings[(rankings["model"] == model) & (rankings["dataset"] == dataset)].copy()

                rrr = rrr.reset_index(drop=True)

                df_real = rrr.sort_values(by=['real'], ascending=False).copy().reset_index(drop=True)
                df_pred = rrr.sort_values(by=['pred'], ascending=False).copy().reset_index(drop=True)

                count = 0
                for i in range(0, len(df_real)):
                    if df_real.loc[i].technique == df_pred.loc[i].technique:
                        count = count + 1

                # count/len(df_real)
                f.write(dataset + "," + model + "," + str(count / len(df_real)) + "," + str(
                    np.array(df_real.technique)).replace('\n', '') + "," + str(np.array(df_pred.technique)).replace(
                    '\n', '') + "," + str(
                    np.array(df_real.real)).replace('\n', '') + "," + str(np.array(df_pred.pred)).replace('\n',
                                                                                                          '') + "," + str(rrr.perc_completeness[0]) + "\n")

    print("Done!")
