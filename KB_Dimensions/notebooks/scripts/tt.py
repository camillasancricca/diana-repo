import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import root_mean_squared_error
from .dataset import get_dataset

original_data = pd.read_json("../kb/KBR.json")

dimensions = original_data.dimension.unique()
models = original_data.model.unique()
datasets = original_data.name.unique()

datasets_fd = ["BachChoralHarmony", "bank", "cancer", "mushrooms", "soybean"]

data_impact = get_dataset()

def training_testing():
    with open("../results/results_prediction_impact.csv", "w") as f:
        f.write("dataset,model,dimension,rmse\n")
        for dataset in datasets:
            for model in models:
                for dimension in dimensions:

                    data = data_impact.copy()

                    if dimension == "consistency" and (dataset in datasets_fd):

                        df = data[(data["model"] == model) & (data["dimension"] == dimension) & (
                                    (data["name"] == "BachChoralHarmony") | (data["name"] == "mushrooms") | (
                                        data["name"] == "bank") | (data["name"] == "cancer") | (
                                                data["name"] == "soybean"))].copy()

                        train = df[df["name"] != dataset]
                        test = df[df["name"] == dataset]

                        columns = df.columns
                        features = columns.drop(
                            ["name", "dimension", "model", "score", "impact", "p_correlated_features_0.5",
                             "p_correlated_features_0.6", "p_correlated_features_0.7", "p_correlated_features_0.8",
                             "p_correlated_features_0.9"])

                        X_train = train[features]
                        y_train = train["impact"]
                        X_test = test[features]
                        y_test = test["impact"]

                        X_train = StandardScaler().fit_transform(X_train)
                        X_train = np.nan_to_num(X_train)

                        X_test = StandardScaler().fit_transform(X_test)
                        X_test = np.nan_to_num(X_test)

                        knn = KNeighborsRegressor(n_neighbors=14, metric='manhattan')
                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(X_test)
                        error = root_mean_squared_error(y_test, y_pred)
                        print(dataset+": "+str(error))
                        f.write(dataset + "," + model + "," + dimension + "," + str(error) + "\n")

                    elif dimension != "consistency":

                        df = data[(data["model"] == model) & (data["dimension"] == dimension)].copy()

                        train = df[df["name"] != dataset]
                        test = df[df["name"] == dataset]

                        columns = df.columns
                        features = columns.drop(
                            ["name", "dimension", "model", "score", "impact", "p_correlated_features_0.5",
                             "p_correlated_features_0.6", "p_correlated_features_0.7", "p_correlated_features_0.8",
                             "p_correlated_features_0.9"])

                        X_train = train[features]
                        y_train = train["impact"]
                        X_test = test[features]
                        y_test = test["impact"]

                        X_train = StandardScaler().fit_transform(X_train)
                        X_train = np.nan_to_num(X_train)

                        X_test = StandardScaler().fit_transform(X_test)
                        X_test = np.nan_to_num(X_test)

                        knn = KNeighborsRegressor(n_neighbors=14, metric='manhattan')
                        knn.fit(X_train, y_train)
                        y_pred = knn.predict(X_test)
                        error = root_mean_squared_error(y_test, y_pred)
                        print(dataset+": "+str(error))
                        f.write(dataset + "," + model + "," + dimension + "," + str(error) + "\n")

    data = pd.read_csv("../results/results_prediction_impact.csv")
    print("Done! Final RMSE: "+str(data.rmse.mean()))




def test_rankings():

    with open("../results/results_ranking_evaluation.csv", "w") as f:
        f.write("dataset,model,dimension,real,pred\n")
        for dataset in datasets:
            for model in models:
                for dimension in dimensions:

                    data = data_impact.copy()

                    if dimension == "consistency" and (dataset in datasets_fd):

                        df = data[(data["model"] == model) & (data["dimension"] == dimension) & (
                                    (data["name"] == "BachChoralHarmony") | (data["name"] == "mushrooms") | (
                                        data["name"] == "bank") | (data["name"] == "cancer") | (
                                                data["name"] == "soybean"))].copy()

                        train = df[df["name"] != dataset]
                        test = df[df["name"] == dataset]

                        columns = df.columns
                        features = columns.drop(
                            ["name", "dimension", "model", "score", "impact", "p_correlated_features_0.5",
                             "p_correlated_features_0.6", "p_correlated_features_0.7", "p_correlated_features_0.8",
                             "p_correlated_features_0.9"])

                        X_train = train[features]
                        y_train = train["impact"]
                        X_test = test[features]
                        y_test = test["impact"]

                        X_train = StandardScaler().fit_transform(X_train)
                        X_train = np.nan_to_num(X_train)

                        X_test = StandardScaler().fit_transform(X_test)
                        X_test = np.nan_to_num(X_test)

                        knn = KNeighborsRegressor(n_neighbors=14, metric='manhattan')
                        knn.fit(X_train, y_train)

                        y_pred = knn.predict(X_test)

                        y_test = y_test.reset_index(drop=True)

                        for i in range(0, len(y_test)):
                            f.write(dataset + "_" + str(i) + "," + model + "," + dimension + "," + str(
                                y_test[i]) + "," + str(y_pred[i]) + "\n")

                    elif dimension != "consistency":

                        df = data[(data["model"] == model) & (data["dimension"] == dimension)].copy()

                        train = df[df["name"] != dataset]
                        test = df[df["name"] == dataset]

                        columns = df.columns
                        features = columns.drop(
                            ["name", "dimension", "model", "score", "impact", "p_correlated_features_0.5",
                             "p_correlated_features_0.6", "p_correlated_features_0.7", "p_correlated_features_0.8",
                             "p_correlated_features_0.9"])

                        X_train = train[features]
                        y_train = train["impact"]
                        X_test = test[features]
                        y_test = test["impact"]

                        X_train = StandardScaler().fit_transform(X_train)
                        X_train = np.nan_to_num(X_train)

                        X_test = StandardScaler().fit_transform(X_test)
                        X_test = np.nan_to_num(X_test)

                        knn = KNeighborsRegressor(n_neighbors=14, metric='manhattan')
                        knn.fit(X_train, y_train)

                        y_pred = knn.predict(X_test)

                        y_test = y_test.reset_index(drop=True)

                        for i in range(0, len(y_test)):
                            f.write(dataset + "_" + str(i) + "," + model + "," + dimension + "," + str(
                                y_test[i]) + "," + str(y_pred[i]) + "\n")

    print("Done!")





def evaluate_rankings():

    rankings = pd.read_csv("../results/results_ranking_evaluation.csv")

    with open("../results/results_ranking_evaluation_total.csv", "w") as f:
        f.write("dataset,model,ranking_eval,ranking_real,ranking_pred,value_real,value_pred\n")
        datasets = rankings.dataset.unique()
        for dataset in datasets:
            for model in models:
                rrr = rankings[(rankings["model"] == model) & (rankings["dataset"] == dataset)].copy()

                rrr = rrr.reset_index(drop=True)

                df_real = rrr.sort_values(by=['real']).copy().reset_index(drop=True)
                df_pred = rrr.sort_values(by=['pred']).copy().reset_index(drop=True)

                count = 0
                for i in range(0, len(df_real)):
                    if df_real.loc[i].dimension == df_pred.loc[i].dimension:
                        count = count + 1

                # count/len(df_real)
                f.write(dataset + "," + model + "," + str(count / len(df_real)) + "," + str(
                    np.array(df_real.dimension)) + "," + str(np.array(df_pred.dimension)) + "," + str(
                    np.array(df_real.real)) + "," + str(np.array(df_pred.pred)) + "\n")

    print("Done!")
