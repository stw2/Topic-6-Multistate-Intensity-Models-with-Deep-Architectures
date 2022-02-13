import numpy as np
import pandas as pd


class TransitionProbaMatrix:
    def __init__(self, nc_proba):
        self.nc_proba = nc_proba  # dictionary of non-competing probability

    def get_competing_proba(self):
        c_proba_dict = {
            'event_01': self.nc_proba['event_01'],
            'event_10': self.nc_proba['event_10'] * (1 - self.nc_proba['event_12']/2),
            'event_12': self.nc_proba['event_12'] * (1 - self.nc_proba['event_10']/2),
            'event_20': self.nc_proba['event_20'] * (1 - (self.nc_proba['event_21']+self.nc_proba['event_23'])/2 +
                                                     self.nc_proba['event_21']*self.nc_proba['event_23']/3),
            'event_21': self.nc_proba['event_21'] * (1 - (self.nc_proba['event_20']+self.nc_proba['event_23'])/2 +
                                                     self.nc_proba['event_20']*self.nc_proba['event_23']/3),
            'event_23': self.nc_proba['event_23'] * (1 - (self.nc_proba['event_20']+self.nc_proba['event_21'])/2 +
                                                     self.nc_proba['event_20']*self.nc_proba['event_21'] / 3)
        }
        return c_proba_dict

    def get_transition_proba(self, length, index):
        """
        :param length: number of observations in the dataframe
        :param index: index of the dataframe
        :return: transition probability "matrix" for each observation
        """
        c_proba_dict = self.get_competing_proba()
        col_zeros = np.zeros(length).reshape(-1, 1)
        transition_proba = np.concatenate(
            (
                1 - c_proba_dict['event_01'], c_proba_dict['event_01'], col_zeros, col_zeros,
                c_proba_dict['event_10'], 1 - c_proba_dict['event_10'] - c_proba_dict['event_12'],
                c_proba_dict['event_12'], col_zeros,
                c_proba_dict['event_20'], c_proba_dict['event_21'],
                1 - c_proba_dict['event_20'] - c_proba_dict['event_21'] - c_proba_dict['event_23'],
                c_proba_dict['event_23'],
                col_zeros, col_zeros, col_zeros, np.ones(length).reshape(-1, 1)
            ), axis=1
        )
        transition_proba_df = pd.DataFrame(transition_proba, index=index)
        return transition_proba_df
