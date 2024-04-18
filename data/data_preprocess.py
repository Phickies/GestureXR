import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


# Declare variable
data_unprocessed_path = './data_collected_unprocessed/'
data_all_frame_path = "./data_all_frame_csv/"


def merge_csv_file(folder_path):
    # List to store individual DataFrame
    df_list = []

    # Iterate over each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Read the CSV file into a DataFrame
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Append the df to the list
            df_list.append(df)

    # Concatenate all DataFrame in the list into one big DataFrame
    return pd.concat(df_list, ignore_index=True)


if __name__ == "__main__":

    # Merge all dataset into one big data
    if not os.path.exists(data_all_frame_path):
        os.makedirs(data_all_frame_path)
        big_data = merge_csv_file(data_unprocessed_path)
        output_path = os.path.join(data_all_frame_path, "data.csv")
        big_data.to_csv(output_path, index=False)
