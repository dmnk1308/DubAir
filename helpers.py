import pandas as pd 
import numpy as np
from re import sub
from scipy.stats import ttest_ind
from scipy.stats import kruskal

# turns all columns (in_cat) into one column (out_cat)
def in_one(df, in_cat, out_cat_name, regex = True, sum = False, drop = True):

    if regex == True:
        # get columns to substitute
        col_filter = df.columns.str.contains(in_cat, case = False, regex = True)
        col_names = df.columns[col_filter]

        # create new column
        if sum == True:
            new_var = df.filter(col_names).sum(axis = 1)
        else:
            new_var = df.filter(col_names).sum(axis = 1)
            new_var[new_var>1] = 1
    
    else:
        col_names = in_cat
        if sum == True:
            new_var = df.filter(col_names).sum(axis = 1)
        else:
            new_var = df.filter(col_names).sum(axis = 1)
            new_var[new_var>1] = 1
    df[out_cat_name] = new_var

    if drop == True:
        df = df.drop(col_names, axis = 1)
    
    return df

# deletes several columns
def drop_col(df, sender, regex = True):
    if regex == True:
        col_filter = df.columns.str.contains(sender, case = False, regex = True)
        sender = df.columns[col_filter] 
    df = df.drop(sender, axis = 1)
    return df

# adds several columns to another existing one
def add_col(df, sender, receiver, regex = True):
    if regex == True:
        col_filter = df.columns.str.contains(sender, case = False, regex = True)
        sender = df.columns[col_filter] 

    df[receiver] = df.filter(sender).sum(axis=1)+df[receiver]
    df = df.drop(sender, axis = 1)
    return df

# Let's define a function that sets up the dummies and drops the category column:
def to_dummy_single(df, X, output_name):
    dummies = pd.get_dummies(df[X])
    dummies.columns = output_name + "_" + dummies.columns

    if len(dummies.columns) == 2:
        dummies = dummies[dummies.columns[0]]

    df = df.join(dummies)
    df = df.drop(X, axis=1)

    return df


def to_dummy(df, X_list, output_names):
    for i in range(len(X_list)):
        df = to_dummy_single(df, X_list[i], output_names[i])

    return df


def clean_comments(text):
    ''' Pre process and convert texts to a list of words
    method inspired by method from eliorc github repo: https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb'''
    text = str(text)
    text = text.lower()

    # Clean the text
    text = sub(r"[^A-Za-z0-9^,!?.\/'+]", " ", text)
    text = sub(r"\+", " plus ", text)
    text = sub(r",", " ", text)
    text = sub(r"\.", " ", text)
    text = sub(r"!", " ! ", text)
    text = sub(r"\?", " ? ", text)
    text = sub(r"'", " ", text)
    text = sub(r":", " : ", text)
    text = sub(r"\s{2,}", " ", text)
    text = sub(r"\s+$", "", text)

    # they were weird
    text = text.replace("br/", "")

    return text


def t_Test(X, y, stats, p_val, names):
    catg = pd.unique(X)
    catg_filter = (X == catg[0])
    sample1 = y[catg_filter]
    sample2 = y[~catg_filter]
    
    t, p = ttest_ind(sample1, sample2, equal_var = False)
    name = X.name

    stats.append(t)
    p_val.append(p)
    names.append(name)


    return t, p

def krus_test(X, y, stats, p_val, names):
    c_list = pd.unique(X)

    F, p = kruskal(*[list(y[X == i]) for i in c_list])
    name = X.name

    stats.append(F)
    p_val.append(p)
    names.append(name)

    return F, p