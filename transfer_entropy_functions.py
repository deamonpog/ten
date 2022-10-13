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

import pandas as pd
import numpy as np
import datetime
import pyinform
import multiprocessing
import typing

# FREQUENCY = 'D'
#
# dummy_data_dir = 'C:\\STUFF\\RESEARCH\\TENet\\DummyData\\'
#
# file_table_1 = 'dummy_news_domain.csv'
# file_table_2 = 'dummy_user_data.csv'
# file_table_3 = 'dummy_social_media_data.csv'
#
# file_table_6 = 'dummy_actor.csv'
# file_table_7 = 'dummy_indv_actor.csv'
# file_table_8 = 'dummy_comm_actor.csv'
# file_table_9 = 'dummy_plat_actor.csv'

# data_dir = 'C:\\STUFF\\RESEARCH\\Brandwatch\\DATA\\MainQuery\\All\\'
# file_table_1 = ''
# file_table_2 = 'users.csv'

# file_table_6 = 'actors.csv'
# file_table_7 = 'indv_actors.csv'
# file_table_8 = 'comm_actors.csv'
# file_table_9 = 'plat_actors.csv'


def get_events_of_actor(actor_id, dataset_df, actors_df, indv_actors_df, comm_actors_df, plat_actors_df):
    """
    Returns the event list of the actor from the dataset.
    :param comm_actors_df: community (or group) actors dataframe with actor_id as index. Defined in Table 8.
    :type comm_actors_df: pd.DataFrame
    :param indv_actors_df: individual actors dataframe with actor_id as index. Defined in Table 7.
    :type indv_actors_df: pd.DataFrame
    :param plat_actors_df: platform actors dataframe with actor_id as index. Defined in Table 9.
    :type plat_actors_df: pd.DataFrame
    :param actors_df: actors dataframe which contains the actor_type. Index should be actor_id. Defined in Table 6.
    :type actors_df: pd.DataFrame
    :param dataset_df: pandas dataframe containing the OSN messages as defined in Table 3.
    :type dataset_df: pd.DataFrame
    :param actor_id: actor_id
    :type actor_id: str
    :return: a pandas dataframe that has same columns as Table 3 but filtered with only the events of the given actor_id
    :rtype: pd.DataFrame
    """
    actor_type = actors_df.loc[actor_id]['actor_type']
    if actor_type == 'plat':
        plat = plat_actors_df.loc[actor_id][0]
        #print(f"{actor_id} is Platform: {plat}")
        return dataset_df[dataset_df['platform'] == plat]
    elif actor_type == 'indv':
        user_id = indv_actors_df.loc[actor_id][0]
        #print(f"{actor_id} is Individual: {user_id}")
        return dataset_df[dataset_df['user_id'] == user_id]
    elif actor_type == 'comm':
        user_list = comm_actors_df.loc[actor_id]['user_id']
        # if community has only one user dont run the code for it 
        # (detected by making sure we get a series for the user_list)
        # in such case we return a datframe with 0 records
        if type(user_list) is pd.Series:
            user_list = user_list.values
            msg = f"{actor_id} is a Community of (size = {len(user_list)}) "
            print_limit = 10
            if len(user_list) <= print_limit:
                msg = f"{msg} : {user_list}"
            else:
                users = ' '.join([f"{u}" for u in np.random.choice(user_list, print_limit, replace=False)])
                msg = f"{msg} : [{users} ...]"
            print(msg)
            return dataset_df[dataset_df['user_id'].isin(user_list)]
        else:
            return dataset_df[0:0] # return 0 records
    else:
        raise Exception(f'Unknown actor type : {actor_id} -> {actor_type}\n{actors_df}')


def generate_timeseries_index(start_time, end_time, frequency):
    """
    Generates an index for a timeseries for a given start time, end time, and a time interval (frequency).
    :param start_time: start time of timeseries
    :type start_time: datetime.date
    :param end_time: end time of timeseries
    :type end_time: datetime.date
    :param frequency: Frequency value as a string. e.g. "12H", "D", "30min"
    :type frequency: str
    :return: generated datetime index
    :rtype: pd.DatetimeIndex
    """
    return pd.DatetimeIndex(pd.date_range(start=start_time, end=end_time, freq=frequency))


def resample_binary_timeseries(timeseries, time_index, frequency):
    """
    Resamples the given timeseries (index must be a datetime) by the given frequency and fills values accordingly
     to match the given time_index series. The value column of the returned series will be binary
     (contain either 0 or 1). 0 says that nothing happened at that time interval, 1 says that something happened in
      that interval.
    :param timeseries: a pandas dataframe with index set to a datetime.
    :type timeseries: pd.DataFrame
    :param time_index: The timeseries index for given frequency. This will be the index of the returned series
    :type time_index: pd.DatetimeIndex
    :param frequency: a string value representing the frequency of resampling. e.g. 'D', '12H', '15min'
    :type frequency: str
    :return: the resampled timeseries with time_index as its index
    :rtype: np.ndarray
    """
    return timeseries.resample(frequency).apply(lambda x: 1 if len(x) > 0 else 0).iloc[:, 0].rename('events').reindex(
        time_index, fill_value=0).values


def multiprocess_resample_actor_binary_timeseries(ordered_actor_id_events_list, time_index, frequency):
    """
    Calculates a binary timeseries for each event list in the given ordered_actor_id_events_list.
     Utilize the Multiprocessing Pools for fast execution over CPUs.
     Results contain an ordered list of binary timeservers of each respective actor event list in the input
     ordered_actor_id_events_list parameter
     (i.e. the order of ordered_actor_id_events_list corresponds to the order of results).
    :param frequency: a string value representing the frequency of resampling. e.g. 'D', '12H', '15min'
    :type frequency: str
    :param ordered_actor_id_events_list: a list that holds the events DataFrame of each actor_id. Order of results
     is correspondent to the order of event lists in this list.
    :type ordered_actor_id_events_list: typing.List[pd.DataFrame]
    :param time_index: a pandas series that will be used as the index for resampling the data
    :type time_index: pd.DatetimeIndex
    :return: a list where each element is an array of binary values (0s and 1s) which represents
     the binary timeseries value that corresponds to each index in the time_index.
    For more information check the resample_binary_timeseries function that is being called by this function.
    :rtype: typing.List[np.ndarray]
    """
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        results = p.starmap(resample_binary_timeseries,
                            [(actor_id_events.set_index('datetime'), time_index, frequency) for actor_id_events in
                             ordered_actor_id_events_list])
    return results


def calculate_te_values(src_actor_id, tgt_actor_id, src_timeseries, tgt_timeseries):
    """
    Calculates the transfer entropy from the given source and target timeseries and returns a list that contains
     [Source, Target, TransferEntropy].
    :param src_actor_id: actor_id of Source
    :type src_actor_id: str
    :param tgt_actor_id: actor_id of Target
    :type tgt_actor_id: str
    :param src_timeseries: binary timeseries of Source
    :type src_timeseries: np.ndarray
    :param tgt_timeseries: binary timeseries of Target
    :type tgt_timeseries: np.ndarray
    :return: the list [Source actor_id, Target actor_id, Transfer Entropy value]
    :rtype: typing.List[str, str, float]
    """
    print(f"[{src_actor_id} ==> {tgt_actor_id}]")
    # print(f"{src_timeseries.shape} ==> {tgt_timeseries.shape}")
    return [src_actor_id, tgt_actor_id, pyinform.transfer_entropy(src_timeseries, tgt_timeseries, 2)]


def multiprocess_run_calculate_te_edge_list(ordered_actor_id_list, ordered_actor_timeseries_list):
    """
    Utilize multiprocessing Pool for calculating all transfer entropy values using the calculate_te_values function.
    Returns a list of calculate_te_values function returns for the provided input parameters.
    :param ordered_actor_id_list: the ordered list of actor_id values.
    :type ordered_actor_id_list: typing.List[str]
    :param ordered_actor_timeseries_list: a list where each element is an array of binary values (0s and 1s)
     which represent the binary timeseries.
    :type ordered_actor_timeseries_list: typing.List[np.ndarray]
    :return: list of actor_id interactions with their corresponding transfer entropy values
    :rtype: typing.List[typing.List[str, str, float]]
    """
    param_list = []
    for src_idx in range(len(ordered_actor_id_list)):
        for tgt_idx in range(len(ordered_actor_id_list)):
            if src_idx == tgt_idx:
                continue
            param_list.append((ordered_actor_id_list[src_idx], ordered_actor_id_list[tgt_idx],
                               ordered_actor_timeseries_list[src_idx], ordered_actor_timeseries_list[tgt_idx]))
    print(f"params ready. Count: {len(param_list)}")
    #with multiprocessing.Pool(multiprocessing.cpu_count() - 1) as p:
    #    results = p.starmap(calculate_te_values, param_list)
    results = []
    for param in param_list:
        r = calculate_te_values(param[0],param[1],param[2],param[3])
        results.append(r)
    print("mult proc done")
    return results


def generate_te_edge_list(actor_id_list, all_events_df, actors_df, indv_actors_df, comm_actors_df, plat_actors_df,
                          frequency):
    """
    Calculates the transfer entropy based edge weights for the given set of actors.
    :param comm_actors_df:
    :type comm_actors_df:
    :param indv_actors_df:
    :type indv_actors_df:
    :param plat_actors_df:
    :type plat_actors_df:
    :param actors_df:
    :type actors_df:
    :param actor_id_list:
    :type actor_id_list:
    :param all_events_df:
    :type all_events_df:
    :param frequency:
    :type frequency:
    :return:
    :rtype:
    """
    # generate time_index
    start_date = all_events_df['datetime'].dt.date.min()
    end_date = all_events_df['datetime'].dt.date.max() + datetime.timedelta(days=1)
    print(f"Data available from {start_date} to {end_date}")
    datetime_index = generate_timeseries_index(start_date, end_date, frequency)
    print("Running resampling timeseries calc...")
    # resample actor timeseries
    actor_timeseries_list = multiprocess_resample_actor_binary_timeseries(
        [get_events_of_actor(actor_id, all_events_df, actors_df, indv_actors_df, comm_actors_df, plat_actors_df) for
         actor_id in actor_id_list],
        datetime_index, frequency)
    print("Running TE edge list calc...")
    print(actor_timeseries_list)
    # calculate te values
    src_tgt_te_list = multiprocess_run_calculate_te_edge_list(actor_id_list, actor_timeseries_list)
    print("Calculation done. Creating dataframe...")
    result_df = pd.DataFrame(src_tgt_te_list, columns=['Source', 'Target', 'TE'])
    return result_df


# def main():
#     # newsdomains = pd.read_csv(dummy_data_dir + file_table_1, index_col='domain_name')
#     # users = pd.read_csv(dummy_data_dir + file_table_2)
#     t3_all_osn_msgs = pd.read_csv(dummy_data_dir + file_table_3, parse_dates=['datetime'])

#     t6_actors = pd.read_csv(dummy_data_dir + file_table_6, index_col='actor_id')
#     t7_indv_actors = pd.read_csv(dummy_data_dir + file_table_7, index_col='actor_id')
#     t8_comm_actors = pd.read_csv(dummy_data_dir + file_table_8, index_col='actor_id')
#     t9_plat_actors = pd.read_csv(dummy_data_dir + file_table_9, index_col='actor_id')

#     # choose actors (e.g. here collects all the actors)
#     selected_indv_actors = t7_indv_actors.index.to_list()
#     selected_comm_actors = t8_comm_actors.index.to_list()
#     selected_plat_actors = t9_plat_actors.index.to_list()
#     actors_of_interest = selected_indv_actors + selected_comm_actors + selected_plat_actors

#     # generate te network for the selected actors
#     results_df = generate_te_edge_list(actors_of_interest, t3_all_osn_msgs, t6_actors, t7_indv_actors, t8_comm_actors,
#                                        t9_plat_actors, '2H')
#     results_df['Weight'] = results_df['TE']
#     results_df.to_csv(dummy_data_dir + 'dummy_actor_te_edges.csv', index=False)


# if __name__ == '__main__':
#     print('--Begin--')
#     main()
#     print('--End--')
