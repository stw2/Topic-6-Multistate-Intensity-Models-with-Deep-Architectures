import numpy as np
import pandas as pd


class ApplyBMModel:
    def __init__(self, df, scaler, var_to_standardize, spline_transformer, splines_names, value_to_fillna,
                 model_coefs, model_intercept):
        self.df = df
        self.scaler = scaler
        self.var_to_standardize = var_to_standardize
        self.spline_transformer = spline_transformer
        self.splines_names = splines_names
        self.value_to_fillna = value_to_fillna
        self.model_coefs = model_coefs
        self.model_intercept = model_intercept

    def replace_na(self):
        df_processed = self.df.fillna(
            value={'fico': self.value_to_fillna[0], 'cltv': self.value_to_fillna[1], 'dti': self.value_to_fillna[2]})
        return df_processed

    def standardize(self, df_processed):
        df_processed[self.var_to_standardize] = self.scaler.transform(df_processed)
        return df_processed

    def generate_splines(self):
        df_splines = pd.DataFrame(
            self.spline_transformer.transform(np.array(self.df['time']).reshape(-1, 1)),
            index=self.df.index, columns=self.splines_names
        )
        return df_splines

    def get_processed_df(self):
        df_processed = self.replace_na()
        df_processed = self.standardize(df_processed)
        df_processed = pd.concat((df_processed, self.generate_splines()), axis=1)
        return df_processed

    def get_non_competing_proba(self):
        df_processed = self.get_processed_df()
        eta = np.matmul(df_processed[self.var_to_standardize + self.splines_names].to_numpy(),
                        self.model_coefs.T) + self.model_intercept
        nc_proba = 1 / (1 + np.exp(-eta))
        if ((nc_proba < 0) | (nc_proba > 1)).sum() > 0:
            raise ValueError('Error in the calculation of non-competing probability')
        df_nc_proba = pd.DataFrame(nc_proba, index=self.df.index, columns=['Y_{ihj}(t)=1'])
        return df_nc_proba