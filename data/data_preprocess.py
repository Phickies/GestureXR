import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

# Declare variable
data_unprocessed_path = './data_collected_unprocessed/'
data_all_frame_path = "./data_all_frame_csv/"

# Config panda
pd.set_option('display.max_columns', None)


def merge_csv_file(folder_path):
    """
    Merge all data collected into one big data frame
    :param folder_path: input folder
    :return: panda DataFrame
    """
    # List to store individual DataFrame
    df_list = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, filename)
            dataframe = pd.read_csv(file_path)

            # Append the df to the list
            df_list.append(dataframe)

    # Concatenate all DataFrame in the list into one big DataFrame
    return pd.concat(df_list, ignore_index=True)


def convert_str_to_int(dataframe):
    """
    Convert the value in the Accel and Gyr from list of string to list of int
    :param dataframe: df need to change
    :return: panda DataFrame
    """
    # Removed invalid data, duplicated data
    dataframe.dropna(inplace=True)
    dataframe.drop_duplicates(inplace=True)

    # Convert string values in lists to integers
    dataframe.Accel = dataframe.Accel.apply(eval)
    dataframe.Gyr = dataframe.Gyr.apply(eval)

    return dataframe


def fix_sep_value(dataframe):
    """
    Fix the duplicate True value, switching back to False.
    :param dataframe: df need to change
    :return: pd DataFrame
    """
    # Find the indices where 'Sep' is True
    true_indices = df.index[df['Sep'] == True]

    # Iterate over the indices and update the value of 'Sep' to False for adjacent True values
    for idx in range(len(true_indices) - 1):
        if true_indices[idx + 1] - true_indices[idx] == 1:
            df.at[true_indices[idx + 1], 'Sep'] = False
    return dataframe


def fix_add_sep_for_pinch2finger(dataframe):
    """
    Fix the dataframe by adding a True sep for each pinch2finger
    :param dataframe: df need to change
    :return: pd DataFrame
    """
    # Find the indices where the label is 'pinch2finger'
    pinch2finger_indices = dataframe.index[dataframe['Label'] == 'pinch2finger']

    # Iterate over the indices and update the value of 'Sep' to True only for the first occurrence
    for idx in pinch2finger_indices:
        # Check if the index is greater than 0 and the previous row's label is not 'pinch2finger'
        if idx > 0 and dataframe.at[idx - 1, 'Label'] != 'pinch2finger':
            dataframe.at[idx, 'Sep'] = True

    return dataframe


if __name__ == "__main__":

    # Merge all dataset into one big data, and preprocessing the data
    print("Merging dataset")
    output_path = os.path.join(data_all_frame_path, "data.csv")
    if not os.path.exists(data_all_frame_path):
        os.makedirs(data_all_frame_path)
        big_data = merge_csv_file(data_unprocessed_path)
        big_data.to_csv(output_path, index=False)

    # Get data
    df = pd.read_csv(output_path)
    print("Converting string to int")
    df = convert_str_to_int(df)
    print("Fixing separating value")
    df = fix_sep_value(df)
    print("Fixing pinch2finger sep value")
    df = fix_add_sep_for_pinch2finger(df)
    print(df.head())
    df.to_csv(output_path, index=False)
