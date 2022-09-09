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
import typing

news_outlets_file = 'C:\\STUFF\\RESEARCH\\TENet\\DummyData\\news_outlets.xlsx - Sheet1.csv'
dummy_data_dir = 'C:\\STUFF\\RESEARCH\\TENet\\DummyData\\'

file_table_1 = 'dummy_news_domain.csv'
file_table_2 = 'dummy_user_data.csv'
file_table_3 = 'dummy_social_media_data.csv'

file_table_5 = 'dummy_gdelt_events.csv'
file_table_6 = 'dummy_actor.csv'
file_table_7 = 'dummy_indv_actor.csv'
file_table_8 = 'dummy_comm_actor.csv'
file_table_9 = 'dummy_plat_actor.csv'


def generate_dummy_table_1():
    # table 1 : news_domain
    domain_list = pd.read_csv(news_outlets_file)['news outlets'].to_list()
    print(dummy_data_dir + file_table_1)
    pd.DataFrame(
        [[domain, np.random.uniform(0, 100), np.random.uniform(0, 100), np.random.choice(['TF', 'TM', 'UF', 'UM'])]
         for
         domain in domain_list]
        , columns=['domain_name', 'trust_score', 'pop_score', 'tufm_class']).to_csv(dummy_data_dir + file_table_1,
                                                                                    index=False)


def generate_dummy_table_2(num_users_per_platform):
    print(dummy_data_dir + file_table_2)
    pd.DataFrame([[idx] + x for idx, x in enumerate(
        [[f"u{i}_{plat}", plat] for plat in ['twitter', 'reddit', 'gab', '4chan'] for i in
         range(num_users_per_platform)])]
                 , columns=['user_id', 'source_user_id', 'platform']).to_csv(dummy_data_dir + file_table_2,
                                                                             index=False)


def generate_dummy_table_3(num_msgs_per_platform):
    print(dummy_data_dir + file_table_3)

    domains_list = pd.read_csv(dummy_data_dir + file_table_1)['domain_name'].values
    users_list = pd.read_csv(dummy_data_dir + file_table_2)
    num_users = users_list.shape[0]
    strftime_format = "%Y-%m-%dT%H:%M:%SZ"
    data_list = []
    msgtime = datetime.datetime(2018, 3, 1)
    msgid = -1
    for i in range(num_msgs_per_platform):
        for plat in users_list['platform'].unique():
            msgtime = msgtime + datetime.timedelta(minutes=np.random.randint(0, 60))
            msgid += 1
            userid = np.random.choice(users_list[users_list['platform'] == plat]['user_id'])
            platform = users_list.iloc[userid]['platform']
            srcmsgid = f"m{i}_{plat}"
            parentid = f"m{np.random.randint(0, i + 1)}_{plat}"
            domain = np.random.choice(domains_list)
            articleurl = f"{domain}/{chr(np.random.randint(ord('a'), ord('a') + 5))}/{np.random.randint(5)}"
            data_list.append(
                [msgid, userid, platform, srcmsgid, parentid, msgtime.strftime(strftime_format), articleurl,
                 domain])
    pd.DataFrame(data_list
                 ,
                 columns=['msg_id', 'user_id', 'platform', 'source_msg_id', 'parent_msg_id', 'time', 'article_url',
                          'domain_name']).to_csv(dummy_data_dir + file_table_3, index=False)


def generate_dummy_table_5(num_events):
    article_urls = pd.read_csv(dummy_data_dir + file_table_3)['article_url'].unique()
    data_list = []
    urlcommidx = 0
    urlcomm = f'guc{urlcommidx}'
    story = f's{urlcommidx}'
    narrative = f'n{urlcommidx}'
    for i in range(num_events):
        data_list.append([f"e{i}", np.random.choice(article_urls), urlcomm, story, narrative])
        if np.random.random() < 0.3:
            urlcommidx += 1
            urlcomm = f'guc{urlcommidx}'
            story = f's{urlcommidx}'
            narrative = f'n{urlcommidx}'
    pd.DataFrame(data_list, columns=['event_id', 'article_url', 'url_community_id', 'story', 'narrative']).to_csv(
        dummy_data_dir + file_table_5, index=False)


def generate_dummy_table_6789(num_indv_actors, num_comm_actors, comm_size_dist_func):
    users_df = pd.read_csv(dummy_data_dir + file_table_2, index_col='user_id')
    all_actors = []
    # platform actors------------------------
    plat_data_list = [[idx, plat] for idx, plat in enumerate(users_df['platform'].unique())]
    plat_actor_df = pd.DataFrame(plat_data_list, columns=['actor_id', 'platform'])
    plat_actor_df.to_csv(dummy_data_dir + file_table_9, index=False)
    all_actors = plat_actor_df.apply(lambda row: [row['actor_id'], 'plat', row['platform']], axis=1).to_list()
    # all_actors = [[aid, 'plat',plat_actor_df.loc[aid]] for aid in plat_actor_df['actor_id'].values]
    print(dummy_data_dir + file_table_9)
    # individual actors----------------------
    next_idx_start = plat_actor_df['actor_id'].max() + 1
    indv_data_list = [[next_idx_start + idx, userid]
                      for idx, userid in enumerate(
            pd.read_csv(dummy_data_dir + file_table_3)['user_id'].value_counts().nlargest(
                num_indv_actors).index.to_list())]
    indv_actor_df = pd.DataFrame(indv_data_list, columns=['actor_id', 'user_id'])
    indv_actor_df.to_csv(dummy_data_dir + file_table_7, index=False)
    print(dummy_data_dir + file_table_7)
    all_actors = all_actors + indv_actor_df.apply(
        lambda row: [row['actor_id'], 'indv', users_df.loc[row['user_id']]['source_user_id']], axis=1).to_list()
    # all_actors = all_actors + [[aid, 'indv'] for aid in indv_actor_df['actor_id'].values]
    # community actors-------------------------
    next_idx_start = indv_actor_df['actor_id'].max() + 1
    comm_actor_list = [[comm_idx + next_idx_start, 'comm', f"comm{comm_idx}"] for comm_idx in range(num_comm_actors)]
    comm_size = [comm_size_dist_func() for comm in comm_actor_list]
    max_comm_size = comm_size[0]
    for s in comm_size:
        if s > max_comm_size:
            max_comm_size = s
    comm_users_draft = np.random.choice(users_df.index.to_list(), (num_comm_actors, max_comm_size), replace=False)
    comm_users = []
    for cidx, comm in enumerate(comm_actor_list):
        for uidx in range(comm_size[cidx]):
            comm_users.append([comm[0], comm_users_draft[cidx][uidx], np.random.random()])
    comm_actor_df = pd.DataFrame(comm_users, columns=['actor_id', 'user_id', 'membership_strength'])
    comm_actor_df.to_csv(dummy_data_dir + file_table_8, index=False)
    print(dummy_data_dir + file_table_8)
    all_actors = all_actors + comm_actor_list
    # all actors--------------------------------
    pd.DataFrame(all_actors
                 , columns=['actor_id', 'actor_type', 'actor_label']).to_csv(dummy_data_dir + file_table_6, index=False)
    print(dummy_data_dir + file_table_6)


def generate_data(num_gdelt_events: int, num_users_per_platform: int, num_msgs_per_platform: int,
                  num_individual_actors: int, num_community_actors: int,
                  community_size_distribution_function: typing.Callable[[], int]):
    """
    Generate all dummy Data files.
    """
    generate_dummy_table_1()
    generate_dummy_table_2(num_users_per_platform)
    generate_dummy_table_3(num_msgs_per_platform)
    generate_dummy_table_6789(num_individual_actors, num_community_actors, community_size_distribution_function)
    generate_dummy_table_5(num_gdelt_events)


if __name__ == "__main__":
    generate_data(num_gdelt_events=100,
                  num_users_per_platform=100,
                  num_msgs_per_platform=1000,
                  num_individual_actors=100,
                  num_community_actors=25,
                  community_size_distribution_function=(lambda: np.random.randint(2, 12)))
