import os
import pandas as pd
import logging

RANK_FILE = 'rank.csv'        # format: User_id \t [ranked item_ids] \t [scores] \t [labels]
GROUP_1_FILE = 'active_test_ratings.txt'
GROUP_2_FILE = 'inactive_test_ratings.txt'


class DataLoader(object):
    def __init__(self, path, sep='\t', seq_sep=',', label='label', rank_file=RANK_FILE, group_1_file=GROUP_1_FILE,
                 group_2_file=GROUP_2_FILE):
        self.rank_df = None
        self.path = path
        self.sep = sep
        self.seq_sep = seq_sep
        self.label = label
        self.rank_file = rank_file
        self.group_1_file = group_1_file
        self.group_2_file = group_2_file
        self._load_data()
        self.g1_df, self.g2_df = self._load_groups()

    def _load_data(self):
        rank_file = os.path.join(self.path, self.rank_file)
        if os.path.exists(rank_file):
            if self.rank_df is None:
                logging.info("load rank csv...")
                self.rank_df = pd.read_csv(rank_file, sep='\t')
                self.rank_df['q'] = 1
                if 'uid' not in self.rank_df:
                    raise ValueError("missing uid in header.")
                logging.info("size of rank file: %d" % len(self.rank_df))
        else:
            raise FileNotFoundError('No rank file found.')

    def _load_groups(self):
        """
        Load advantaged/disadvantaged group info file and split the all data dataframe
        into two group-dataframes
        :return: group 1 dataframe (advantaged), group 2 dataframe (disadvantaged)
        """
        if self.rank_df is None:
            self._load_data()
        group_1_file = os.path.join(self.path, self.group_1_file)
        group_2_file = os.path.join(self.path, self.group_2_file)
        if os.path.exists(group_1_file):
            logging.info("load group 1 info txt...")
            g1_df = pd.read_csv(group_1_file, sep='\t')
        else:
            raise FileNotFoundError('No Group 1 file found.')

        if os.path.exists(group_2_file):
            logging.info("load group 2 info txt...")
            g2_df = pd.read_csv(group_2_file, sep='\t')
        else:
            raise FileNotFoundError('No Group 2 file found.')

        if 'uid' in g1_df and 'uid' in g2_df:
            g1_user_list = list(set(g1_df['uid'].tolist()))
            g2_user_list = list(set(g2_df['uid'].tolist()))
        else:
            raise ValueError('No uid found in the group dataframe.')

        group_1_df = self.rank_df[self.rank_df['uid'].isin(g1_user_list)]
        group_2_df = self.rank_df[self.rank_df['uid'].isin(g2_user_list)]
        return group_1_df, group_2_df