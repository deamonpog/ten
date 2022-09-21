"""
Generating Transfer Entropy Network and Retweet Network from the twitter data.

Author:
Chathura Jayalath
Complex Adaptive Systems Laboratory, UCF
"""

"""
MIT License

Copyright (c) 2022 Chathura Jeewaka Jayalath Don Dimungu Arachchige (deamonpog)
                    Complex Adaptive Systems Laboratory
                    University of Central Florida

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import glob
import os
import pandas as pd
import pyinform
import time
import multiprocessing
import pprint
import numpy as np
import datetime
from typing import List

DATA_FILE_DIRECTORY = 'C:\\STUFF\\RESEARCH\\TENet\\DATA'
MINIMUM_POSTS = {'#wildfire': 0, '#infinitychallenge': 0}
FREQUENCY = '4H'
SORTED_HASHTAGS_LIST = ['#wildfire', '#infinitychallenge']
OUTPUT_DIRECTORY = 'C:\\STUFF\\RESEARCH\\TENet\\OUTPUTS'
UNIQUE_NOT_FOUND_STRING = '__NOT_FOUND__'
THRESHOLD_TE = 0.00


def read_csv_data(data_directory):
    """
    Reads data from all the csv files in the given directory
    :param data_directory: Path to the directory that contains the csv files
    :type data_directory: str
    :return: pandas Dataframe that contains all the data from all csv files
    :rtype: pd.Dataframe
    """
    data_files = glob.glob(os.path.join(data_directory, "*.csv*"))
    print(data_files)
    df_list = []
    for idx, file in enumerate(data_files):
        print(f"Reading {idx + 1} of {len(data_files)} files.\nFile name: {file}")
        df = pd.read_csv(data_files[0], skiprows=6, parse_dates=['Date'], dtype={'Twitter Author ID': str})
        df = df[['Date', 'Hashtags', 'Twitter Author ID', 'Author', 'Url', 'Thread Id', 'Thread Author']]
        df = df.rename(columns={'Twitter Author ID': 'AuthorID',
                                'Thread Id': 'ThreadId', 'Thread Author': 'ThreadAuthor'})
        df_list.append(df)
    result_df = pd.concat(df_list).drop_duplicates()
    return result_df


def generate_timeseries_index(start_time, end_time, frequency):
    """
    Generates an index for a timeseries for a given start time, end time, and a time interval (frequency).
    :param start_time: start time of timeseries
    :type start_time: datetime
    :param end_time: end time of timeseries
    :type end_time: datetime
    :param frequency: Frequency value as a string. e.g. "12H", "D", "30min"
    :type frequency: str
    :return:
    :rtype:
    """
    return pd.DatetimeIndex(pd.date_range(start=start_time, end=end_time, freq=frequency))


def generate_sampled_binary_timeseries(timeseries, time_index, frequency='D'):
    """
    Resamples the given timeseries (index must be datetime) by the given frequency and fills values accordingly to match
     the given time_index series. The value column of the returned series will be binary (contain either 0 or 1).
     0 says that nothing happened at that time interval, 1 says that something happened in that interval.
    :param timeseries: a pandas dataframe with index set to a datetime.
    :type timeseries: pd.Dataframe
    :param time_index: The timeseries index for given frequency
    :type time_index: pd.DatetimeIndex
    :param frequency: a string value representing the frequncy of resampling. e.g. 'D', '12H', '15min'
    :type frequency: str
    :return: the resampled timeseries with time_index as its index
    :rtype: pd.Series
    """
    result = timeseries.resample(frequency).apply(lambda x: 1 if len(x) > 0 else 0).iloc[:, 0].rename('events').reindex(
        time_index, fill_value=0)
    # print('\n\t-----timeseris: ', timeseries, '\n\t----result', result)
    return result


def calculate_transfer_entropy(source: list, target: list, k: int) -> float:
    """
    Calculates the transfer entropy between two given timeseries, source and target.
    :param source: List of binary values representing the timeseries of source
    :type source: list
    :param target: List of binary values representing the timeseries of target
    :type target: list
    :param k: History length
    :type k: int
    :return: Transfer entropy from source to target
    :rtype: float
    """
    return pyinform.transfer_entropy(source, target, k=k)


def calculate_author_values(author_id, num_mentions, all_data, time_index, frequency):
    """
    Returns the Node information and generated timeseries of the Author
    Reads the data from ALL_DATA global variable which must have been populated before invoking this function.
    :param time_index: The timeseries index for given frequency
    :type time_index: pd.DatetimeIndex
    :param all_data: Dataframe of all events
    :type all_data: pd.Dataframe
    :param num_mentions:
    :type num_mentions: int
    :param author_id: AuthorID
    :type author_id: str
    :return: A tuple of two items. 0th item is the node info that contains [author_id, author, number of mentions]. 1st
     item is the timeseries values as a binary list.
    :rtype: tuple
    """
    author_id_events = all_data[all_data['AuthorID'] == author_id]
    return ([author_id, author_id_events['Author'].iloc[0], num_mentions],
            generate_sampled_binary_timeseries(author_id_events.set_index('Date'), time_index, frequency).values)


def multiprocess_run_calculate_author_values(author_list, author_id_to_num_mentions_dict, all_data, time_index,
                                             frequency):
    """
    Calculates the author node and timeseries data for each author in the given author_list using Multiprocessing Pools.
    Results contain an ordered list of results which represent the data for each respective author in the order of
     author_list parameter. Each node will correspond to an author_id.
    :param author_list: list of author_id values for which we calculate nodes.
    :type author_list: list
    :param author_id_to_num_mentions_dict: a dictionary that contains number of mentions value for each author_id
    :type author_id_to_num_mentions_dict: dict
    :param all_data: pandas dataframe that contains all the data for processing
    :type all_data: pd.Dataframe
    :param time_index: a pandas series that will be used as the index for resampling the data
    :type time_index: pd.Series
    :return: a list where each element represents a tuple (of length = 2) for corresponding author_id.
    Each element is a tuple of two items. 0th item is the node info that contains
     [author_id, author_username, number of mentions]. 1st item is the timeseries values as a binary list.
    For more information check the calculate_author_values function that is being called by this function.
    :rtype: list
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = p.starmap(calculate_author_values,
                            [(author, author_id_to_num_mentions_dict[author], all_data, time_index, frequency) for
                             author in
                             author_list])
    return results


def calculate_te_values(src_idx, tgt_idx, time_series_list):
    return [src_idx, tgt_idx,
            calculate_transfer_entropy(time_series_list[src_idx], time_series_list[tgt_idx], 2)]


def multiprocess_run_calculate_te_edge_list(enumerated_author_list, time_series_list):
    param_list = []
    for src_idx, src_author in enumerated_author_list:
        for tgt_idx, tgt_author in enumerated_author_list:
            if src_author == tgt_author:
                continue
            param_list.append((src_idx, tgt_idx, time_series_list))
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = p.starmap(calculate_te_values, param_list)
    return [[enumerated_author_list[r[0]][1], enumerated_author_list[r[1]][1], r[2]] for r in results]


def calculate_retweet_network(data_df, selected_author_ids):
    """
    Calculates the retweet network, given the dataframe of original data.
    Following fields are required by this function : ['Author','ThreadAuthor']
    :param selected_author_ids: a list of author_ids that are to be included in the network, others are removed
    :type selected_author_ids: list
    :param data_df: pandas dataframe containing the required two columns
    :type data_df: pd.Dataframe
    :return: returns a dataframe that contain ['Source', 'Target', 'Count] columns. Source and Target columns are
     author_id values. Count column contains the number of times this relationship was found within the input dataframe.
    :rtype: pd.Dataframe
    """
    author_id_set = set(selected_author_ids)
    src_tgt_count = data_df.groupby(
        ['ThreadAuthorID', 'AuthorID'], as_index=False).size()
    src_tgt_count.rename(columns={'ThreadAuthorID': 'Source', 'AuthorID': 'Target', 'size': 'Count'}, inplace=True)
    src_tgt_count = src_tgt_count[src_tgt_count['Target'].isin(author_id_set)][['Source', 'Target', 'Count']]
    return src_tgt_count


def find_hashtag(hashtag_list_str: str, sorted_target_hashtags: List[str]) -> object:
    if type(hashtag_list_str) != str:
        return None
    hashtag_list = hashtag_list_str.split(", ")
    for lfh in sorted_target_hashtags:
        if lfh in hashtag_list:
            return lfh
    return None


def analyze_data(data_df, minimum_posts, time_index_series, frequency, title):
    task_start_time = time.time()

    # find high activity authors
    author_id_to_num_mentions_dict = data_df['AuthorID'].value_counts().to_dict()
    high_activity_author_ids = [a for a in author_id_to_num_mentions_dict if
                                author_id_to_num_mentions_dict[a] > minimum_posts]
    indexed_high_activity_author_ids = [(i, j) for i, j in enumerate(high_activity_author_ids)]
    print(f"High Activity Users : (Count: {len(high_activity_author_ids)})")
    pprint.pprint(np.array(high_activity_author_ids[:30]))

    print(f"{time.time() - task_start_time} seconds spent on reading data and calculating activity.")
    task_start_time = time.time()

    # calculate retweet network
    src_tgt_count_df = calculate_retweet_network(data_df, high_activity_author_ids)

    print(f"{time.time() - task_start_time} seconds spent on generating retweet network.")
    task_start_time = time.time()

    all_author_ids_in_retweet_network = set(src_tgt_count_df['Source'].to_list())
    all_author_ids_in_retweet_network.update(set(src_tgt_count_df['Target'].to_list()))
    print(f"Authors Count in Retweet network : {len(all_author_ids_in_retweet_network)}")

    # write retweet network to file
    src_tgt_count_df['Weight'] = src_tgt_count_df['Count']
    retweet_edges_file = os.path.join(OUTPUT_DIRECTORY, f'{title}_retweet_edges.csv')
    print(src_tgt_count_df)
    src_tgt_count_df.to_csv(retweet_edges_file, index=False)
    del src_tgt_count_df
    print(f"Retweet network written to : {retweet_edges_file}")

    print(f"{time.time() - task_start_time} seconds spent on writing retweet network to file.")
    task_start_time = time.time()

    # calculate author data
    results = multiprocess_run_calculate_author_values(high_activity_author_ids, author_id_to_num_mentions_dict,
                                                       data_df,
                                                       time_index_series, frequency)
    # pprint.pprint(results)
    time_series_list = [r[1] for r in results]
    node_list = [r[0] for r in results]
    del results

    print(f"{time.time() - task_start_time} seconds spent on resampling the timeseries and setting up node data")
    task_start_time = time.time()

    print("Time series list: ", len(time_series_list))
    print("Node list: ", len(node_list))

    # generate node list file for Gephi
    node_list_df = pd.DataFrame(node_list, columns=['Id', 'Label', 'Count'])
    node_list_file_name = os.path.join(OUTPUT_DIRECTORY, f'{title}_nodes.csv')
    node_list_df.to_csv(node_list_file_name, index=False)
    print(f'Node list written to : {node_list_file_name}')
    print(f"Node file : (Nodes/Lines : {node_list_df.shape[0]})")
    print(node_list_df)
    del node_list_df
    del node_list

    print(f"{time.time() - task_start_time} seconds spent on writing node list to file.")
    task_start_time = time.time()

    # the te values calculated for enumerated authors
    src_tgt_te = multiprocess_run_calculate_te_edge_list(indexed_high_activity_author_ids, time_series_list)

    print(f"{time.time() - task_start_time} seconds spent on calculating transfer entropy.")
    task_start_time = time.time()

    # generate edge list file for Gephi
    edge_list_df = pd.DataFrame(src_tgt_te, columns=['Source', 'Target', 'TE'])
    print("Unique Values of TE : ", edge_list_df.TE.unique())
    edge_list_df['Weight'] = edge_list_df['TE'].apply(lambda x: 1 if x > THRESHOLD_TE else 0)
    max_te = edge_list_df['TE'].max()
    edge_list_df['Normalized'] = edge_list_df['TE'].apply(lambda x: x / max_te)
    print(edge_list_df)
    te_edges_file = os.path.join(OUTPUT_DIRECTORY, f'{title}_te_edges.csv')
    print(f"Transfer Entropy edges file written to : {te_edges_file}")
    edge_list_df.to_csv(te_edges_file, index=False)

    print(f"{time.time() - task_start_time} seconds spent on writing edge list to file.")


def main(start_date, end_date):
    """
    :param start_date: inclusive
    :type start_date: datetime.datetime
    :param end_date: inclusive
    :type end_date: datetime.datetime
    """
    # Read all files and populate the global variables
    all_data_df = read_csv_data(DATA_FILE_DIRECTORY)
    all_data_df = all_data_df[(start_date <= all_data_df['Date']) & (all_data_df['Date'] <= end_date)]

    # verify that UNIQUE_NOT_FOUND_STRING is not in the dataset
    found = all_data_df[all_data_df['Author'] == UNIQUE_NOT_FOUND_STRING].shape[0] > 0
    print(f"Is '{UNIQUE_NOT_FOUND_STRING}' an existing username? {found} ")
    if found:
        raise Exception(f"'{UNIQUE_NOT_FOUND_STRING}' is existing in the dataset. Please choose some other string.")

    # add Hashtag column
    all_data_df['Hashtag'] = all_data_df["Hashtags"].apply(lambda x: find_hashtag(x, SORTED_HASHTAGS_LIST))
    all_data_df.drop(columns=["Hashtags"], inplace=True)

    # add ThreadAuthorID column
    author_to_author_id = all_data_df.set_index('Author')['AuthorID'].to_dict()
    all_data_df['ThreadAuthorID'] = all_data_df['ThreadAuthor'].apply(
        lambda x: author_to_author_id[x] if x in author_to_author_id else UNIQUE_NOT_FOUND_STRING)

    print(all_data_df.head())

    # get start date and end date and create time index for the timeseries
    # start_date = all_data_df['Date'].dt.date.min()
    # end_date = all_data_df['Date'].dt.date.max() + datetime.timedelta(days=1)
    print(f"Data available from {all_data_df['Date'].min()} to {all_data_df['Date'].max()}")
    time_index_series = generate_timeseries_index(start_date, end_date, FREQUENCY)
    print("Time index: ")
    pprint.pprint(time_index_series)

    # filter data to include only the valid author_id values
    all_data_df = all_data_df[(all_data_df['AuthorID'] != UNIQUE_NOT_FOUND_STRING)]

    for hashtag in SORTED_HASHTAGS_LIST:
        print(f"\n\t******* Processing the hashtag: {hashtag}")
        filtered_data_df = all_data_df[all_data_df['Hashtag'] == hashtag]
        analyze_data(filtered_data_df, MINIMUM_POSTS[hashtag], time_index_series, FREQUENCY, hashtag[1:])
        print(f"\t------- Finished the hashtag: {hashtag}")


if __name__ == '__main__':
    print("-- Parameters --")
    print("Sorted Hashtags List:")
    pprint.pprint(SORTED_HASHTAGS_LIST)
    print("Minimum Posts:")
    pprint.pprint(MINIMUM_POSTS)
    print(f"Frequency: {FREQUENCY}")
    print(f"Threshold TE: {THRESHOLD_TE}")
    print(f"Unique Not Found Str: \"{UNIQUE_NOT_FOUND_STRING}\"")
    print("--Begin--")
    main(datetime.datetime(2021, 12, 1, 0, 0, 0), datetime.datetime(2021, 12, 31, 23, 59, 59))
    print("--End--")
