import pandas as pd

def get_dataset():

    data = pd.read_json("../kb/KBR.json")

    models = data.model.unique()
    datasets = data.name.unique()

    data = pd.read_json("../kb/KBR.json")
    data_original = pd.read_json("../kb/KBR_original.json")
    data["impact"] = 0
    for dataset in datasets:
        for model in models:
            new_data = data[(data["name"] == dataset) & (data["model"] == model)]["score"] / data_original[(data_original["name"] == dataset) & (data_original["model"] == model)][
                           "score"].values[0]
            data.loc[(data["name"] == dataset) & (data["model"] == model), "impact"] = new_data

    return data

#if __name__ == '__main__':
#    df = get_dataset()
#    df.to_csv('KBR.csv',index=None)
