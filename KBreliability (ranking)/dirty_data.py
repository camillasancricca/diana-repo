import numpy as np
from numpy import random

def check_datatypes(df):
    for col in df.columns:
        if (df[col].dtype == "bool"):
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return df


def out_of_range(std, maximum, minimum, std_coeff):
    std = std * std_coeff
    limit_h1 = maximum + std
    limit_h2 = minimum - std
    limit_m1 = limit_h1 + std
    limit_m2 = limit_h2 - std
    limit_e1 = limit_m1 + std
    limit_e2 = limit_m2 - std

    foo = ["easy", "medium", "hard"]
    foo1 = ["up", "down"]
    f = random.choice(foo)
    f1 = random.choice(foo1)

    if f == "easy":
        n = 'e'
        if f1 == "up":
            number = np.random.uniform(limit_m1, limit_e1)
        else:
            number = np.random.uniform(limit_e2, limit_m2)

    if f == "medium":
        n = 'm'
        if f1 == "up":
            number = np.random.uniform(limit_h1, limit_m1)
        else:
            number = np.random.uniform(limit_m2, limit_h2)

    if f == "hard":
        n = 'h'
        if f1 == "up":
            number = np.random.uniform(maximum + 1, limit_h1)
        else:
            number = np.random.uniform(limit_h2, minimum - 1)
    return number, n


def injection(dataset, name_class, p, std_coeff, seed):

    np.random.seed(seed)
    df_dirt = dataset.copy()

    comp = [p, (1-p)/2, (1-p)/2]
    df_dirt = check_datatypes(df_dirt)

    for col in df_dirt.columns:

        if col != name_class:

            if df_dirt[col].dtype != "object":

                std = float(np.std(df_dirt[col]))
                rand = np.random.choice([None, True, False], size=df_dirt.shape[0], p=comp)
                df_dirt.loc[rand == True, col] = np.nan

                selected = df_dirt.loc[rand == False, col]
                t = 0
                for i in selected:
                    minimum = float(df_dirt[col].min())
                    maximum = float(df_dirt[col].max())
                    selected.iloc[t:t + 1], type = out_of_range(std, maximum, minimum, std_coeff)
                    t += 1
                df_dirt.loc[rand == False, col] = selected

            else:
                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=[1-p,p])
                df_dirt.loc[rand == True, col] = np.nan

    print("saved dirty dataset {}%".format(round((p) * 100)))
    return df_dirt

