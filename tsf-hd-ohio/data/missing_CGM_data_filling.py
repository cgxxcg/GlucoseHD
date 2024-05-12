import pandas as pd
import numpy as np
import math


def remove_nan_strat_end(df, attr):
    index_start = 0
    index_end = 0
    if attr in list(df):
        for i in range(len(df) - 1):
            if np.isnan(df[attr][i]) == False:
                index_start = i
                break
        for j in range(len(df) - 1, 0, -1):
            if np.isnan(df[attr][j]) == False:
                index_end = j + 1
                break
        df = df[index_start:index_end]
        df = df.reset_index(drop=True)
        print("Preprocessing: Remove nan successfully!")
        return df
    else:
        print('Unable to remove: No Attibutes Found\n')
        return 0


# fill in nan values with "1"
# TODO: use average value to fill in missing CGMs 
def filling_CGM(testpath):
    filepath = "../dataset/processedcsv/ohio540.csv"
    a_test = pd.read_csv(filepath, usecols=['CGM'])
    a_test = remove_nan_strat_end(a_test, 'CGM')  
    AA = a_test['CGM'].fillna(1)
    return AA