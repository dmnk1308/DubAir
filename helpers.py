import pandas as pd 
import numpy as np

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