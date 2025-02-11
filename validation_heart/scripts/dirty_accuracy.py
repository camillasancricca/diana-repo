import numpy as np
import numpy.random
import random

def out_of_range(std, maximum, minimum, std_coeff):
    std = std*std_coeff
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
            number = numpy.random.uniform(limit_m1, limit_e1)
        else: 
            number = numpy.random.uniform(limit_e2, limit_m2)
            
    if f == "medium":
        n = 'm'
        if f1 == "up":
            number = numpy.random.uniform(limit_h1, limit_m1)
        else: 
            number = numpy.random.uniform(limit_m2, limit_h2)

    if f == "hard":
        n = 'h'
        if f1 == "up":
            number = numpy.random.uniform(maximum+1, limit_h1)
        else: 
            number = numpy.random.uniform(limit_h2, minimum - 1)
    return number, n


def check_datatypes(df):
    for col in df.columns:
        if df[col].dtype == "bool":
            df[col] = df[col].astype('string')
            df[col] = df[col].astype('object')
    return  df

#genera la lista di chiavi del dataframe
def get_names(df, name_class):
    l=[]
    tmp = df
    tmp.drop(columns=[name_class])
    for i in tmp.keys():
        l.append(i)
    l.pop()
    return l


def injection(df_pandas, name_class, p, std_coeff, seed):

    np.random.seed(seed)

    df_dirt = df_pandas.copy()
    comp = [p,1-p]
    df_dirt = check_datatypes(df_dirt)

    for col in df_dirt.columns:
        if col != name_class:

            if df_dirt[col].dtype != "object":
                std = float(np.std(df_dirt[col]))
                rand = np.random.choice([True, False], size=df_dirt.shape[0], p=comp)
                selected = df_dirt.loc[rand == True,col]

                list_index=[]
                list_type=[]
                list_value=[]
                list_index.append(col)
                list_value.append(col)
                list_type.append(col)
                t=0
                for i in selected:
                    minimum = float(df_dirt[col].min())
                    maximum = float(df_dirt[col].max())
                    selected.iloc[t:t+1], type = out_of_range(std, maximum, minimum, std_coeff)
                    index = selected.iloc[t:t+1].index[0]
                    value= selected.iloc[t:t+1].values.tolist()[0]
                    list_index.append(index)
                    list_type.append(type)
                    list_value.append(value)
                    t+=1
                df_dirt.loc[rand == True,col]=selected

    print("saved accuracy{}%".format(round((1-p)*100)))
        
    return df_dirt


def fill_list(list_index, list_type, list_value, df):
    
    list_index.pop(0)
    list_value.pop(0)
    list_type.pop(0)
    final_index=[]
    final_value=[]
    final_type=[]
    for i in range(0,len(df)):
        if i in list_index:
            final_type.append(list_value[list_index.index(i)])
            final_value.append(list_type[list_index.index(i)])
            final_index.append(i)
        else: 
            final_index.append(0)
            final_value.append(0)
            final_type.append(0)
    return final_value, final_type

def fill_listA(list, len):
    
    
    final_value=[]
    for i in range(0,len):
        if i in list:
            final_value.append(1)
        else:    
            final_value.append(0)
    return final_value
