"""
Main execution file for calculating Transfer Entropy Network

Author:
Chathura Jayalath
Complex Adaptive Systems Laboratory, UCF
"""

import glob
import pandas as pd
import pyinform
import pprint
import time
import multiprocessing

DATA_FILE_DIRECTORY: str = 'C:\\STUFF\\RESEARCH\\TENet\\DATA\\*'
MINIMUM_POSTS = 10
FREQUENCY = 'D'


def read_data(data_directory):
    data_files = glob.glob(data_directory)
    print(data_files)
    df_list = []
    for idx, file in enumerate(data_files):
        print(f"Reading {idx + 1} of {len(data_files)} files.\nFile name: {file}")
        df = pd.read_csv(data_files[0], skiprows=6, parse_dates=['Date'])
        df = df[['Date', 'Twitter Author ID', 'Author', 'Url']]
        df = df.rename(columns={'Twitter Author ID': 'AuthorID'})
        df_list.append(df)
    return pd.concat(df_list).drop_duplicates()


def generate_timeseries_index(start_time, end_time, frequency):
    return pd.DatetimeIndex(pd.date_range(start=start_time, end=end_time, freq=frequency))


def generate_sampled_binary_timeseries(timeseries, time_index, frequency='D'):
    return timeseries.resample(frequency).apply(lambda x: 1 if len(x) > 0 else 0).iloc[:, 0].rename('events').reindex(
        time_index, fill_value=0)


def calculate_transfer_entropy(source: list, target: list, k: int) -> float:
    return pyinform.transfer_entropy(source, target, k=k)


def calculate_author_values(author_id, num_mentions, all_data, time_index):
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
    :return: A tuple of two items. 0th item is the node info. 1st item is the timeseries as a binary list.
    :rtype: tuple
    """
    global FREQUENCY
    author_events = all_data[all_data['AuthorID'] == author_id]
    return ([author_id, author_events['Author'].iloc[0], num_mentions],
            generate_sampled_binary_timeseries(author_events.set_index('Date'), time_index, FREQUENCY).values)


def multiprocess_run_calculate_author_values(author_list, num_mentions_dict, all_data, time_index):
    results = None
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = p.starmap(calculate_author_values,
                            [(author, num_mentions_dict[author], all_data, time_index) for author in author_list])
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
    results = None
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = p.starmap(calculate_te_values, param_list)
    return [[enumerated_author_list[r[0]][1], enumerated_author_list[r[1]][1], r[2]] for r in results]


def main():
    task_start_time = time.time()

    # Read all files and populate the global variables
    all_data_df = read_data(DATA_FILE_DIRECTORY)
    time_index_series = generate_timeseries_index(all_data_df['Date'].dt.date.min(), all_data_df['Date'].dt.date.max(),
                                                  FREQUENCY)

    # fight high activity authors
    author_vs_num_records = all_data_df['AuthorID'].value_counts().to_dict()
    high_activity_authors = [a for a in author_vs_num_records if author_vs_num_records[a] > MINIMUM_POSTS]
    enumerated_high_activity_authors = [(i, j) for i, j in enumerate(high_activity_authors)]
    print(high_activity_authors)

    print(f"{time.time() - task_start_time} seconds spent on reading data and calculating activity.")
    task_start_time = time.time()

    # calculate author data
    results = multiprocess_run_calculate_author_values(high_activity_authors, author_vs_num_records, all_data_df,
                                                       time_index_series)
    time_series_list = [r[1] for r in results]
    node_list = [r[0] for r in results]
    del results

    print(f"{time.time() - task_start_time} seconds spent on resampling the timeseries and setting up node data")
    task_start_time = time.time()

    print("Time series list: ", len(time_series_list))
    print("Node list: ", len(node_list))

    # generate node list file for Gephi
    node_list_df = pd.DataFrame(node_list, columns=['Id', 'Label', 'Count'])
    node_list_df.to_csv('nodes.csv', index=False)
    del node_list_df
    del node_list

    print(f"{time.time() - task_start_time} seconds spent on writing node list to file.")
    task_start_time = time.time()

    # the te values calculated for enumerated authors
    src_tgt_te = multiprocess_run_calculate_te_edge_list(enumerated_high_activity_authors, time_series_list)

    print(f"{time.time() - task_start_time} seconds spent on calculating transfer entropy.")
    task_start_time = time.time()

    # generate edge list file for Gephi
    edge_list_df = pd.DataFrame(src_tgt_te, columns=['Source', 'Target', 'TE'])
    print(edge_list_df)
    edge_list_df['Weight'] = edge_list_df['TE'].apply(lambda x: 1 if x > 0.05 else 0)
    max_te = edge_list_df['TE'].max()
    edge_list_df['Normalized'] = edge_list_df['TE'].apply(lambda x: x / max_te)
    print(edge_list_df)
    edge_list_df.to_csv('edges.csv', index=False)

    print(f"{time.time() - task_start_time} seconds spent on writing edge list to file.")
    task_start_time = time.time()


if __name__ == '__main__':
    print("--Begin--")
    main()
    print("--End--")
