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
        return df
    else:
        print('Unable to remove: No Attibutes Found\n')
        return 0


# fill in nan values with 
def filling_CGM(df_data) -> pd.Series:  
    """
    Fill NaN values in the 'CGM' column of the DataFrame with the average of nearby non-null values.

    Parameters:
        df_data (pd.DataFrame): Input DataFrame containing 'Time' and 'CGM' columns.

    Returns:
        pd.Series: Series containing the cleaned 'CGM' column data.
    """
    df_copy = df_data.copy() # avoid modifying original value
    df_copy = remove_nan_strat_end(df_copy, 'CGM')  
    df_copy['Time'] = pd.to_datetime(df_copy['Time'], format='%d-%b-%Y %H:%M:%S')
    
    for index, row in df_copy.iterrows():
        if pd.isna(row['CGM']):
            # Extract hour and minute from the current time
            hour_minute = row['Time'].strftime('%H:%M')

            # Find rows with non-null CGM values matching the same hour and minute
            matched_rows = df_copy[(~df_copy['CGM'].isna()) & (df_copy['Time'].dt.strftime('%H:%M') == hour_minute)]

            if len(matched_rows) >= 3:
                # Select the current row and the next two consecutive rows
                selected_rows = matched_rows.iloc[:3]

                # Calculate the average of the CGM values
                avg_cgm_value = selected_rows['CGM'].mean()

                # Replace the NaN CGM value with the calculated average
                df_copy.at[index, 'CGM'] = avg_cgm_value
 
    if df_copy['CGM'].isna().sum() == 0:
        print("Remove NaN successfully")
    else:
        print("Remove NaN failed")
        return 
    
    return df_copy['CGM']
    
    
    
    
    # AA = df_copy['CGM'].fillna(150)
    # return AA