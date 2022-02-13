""" Module docstring """

from copy import deepcopy
import warnings
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
import numpy as np
import pandas as pd
from scipy.special import expit
from sddr import Sddr
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import roc_auc_score
from torch import nn
from torch import optim
from .CustomNNModule import CustomeNNModule
from .PrepareEmbedding import PrepareEmbedding

class SddrEstimator(BaseEstimator, ClassifierMixin):

    """ Class docstring """

    def __init__(self,
        distribution='Bernoulli',
        spline_num_knots=4,
        spline_degree=3,
        spline_lambda_df_diff=0,
        num_hidden_layers=1,
        layer_size_factor=1,
        activation=nn.ReLU(),
        output_size_factor=1,
        batch_size=20000,
        epochs=100,
        optimizer=optim.RMSprop,
        learning_rate=1e-2,
        momentum=0.1,
        val_split=0.2,
        early_stop_epochs=10,
        early_stop_epsilon=1e-4,
        dropout_rate=0.01,
        output_dir=None) -> None:

        self.distribution = distribution
        self.spline_num_knots = spline_num_knots
        self.spline_degree = spline_degree
        self.spline_lambda_df_diff = spline_lambda_df_diff
        self.num_hidden_layers = num_hidden_layers
        self.layer_size_factor = layer_size_factor
        self.activation = activation
        self.output_size_factor = output_size_factor
        self.batch_size = batch_size
        self.epochs = epochs
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.val_split = val_split
        self.early_stop_epochs = early_stop_epochs
        self.early_stop_epsilon = early_stop_epsilon
        self.dropout_rate = dropout_rate
        self.output_dir = output_dir

    def fit(self, X, y, plot=False, seed=None):

        """ Method docstring """

        # set useful attributes
        self.train_X_ = deepcopy(X)
        self.train_y_ = deepcopy(y)
        if self.train_X_.shape[0] != self.train_y_.shape[0]:
            raise ValueError(f"X and y have different number of rows. X:{self.train_X_.shape}, y:{self.train_y_.shape}")
        self.var_n_ = ['fico', 'mi_pct', 'cnt_units', 'cltv', 'dti', 'current_upb', 'int_rt', 'orig_loan_term']
        self.var_c2_ = ['flag_fthb', 'cnt_borr', 'flag_sc', 'program_ind']
        self.var_c3_ = ['occpy_sts', 'channel', 'prop_type', 'loan_purpose', 'seller_name', 'servicer_name', 'pre_harp', 'cd_msa']
        self.classes_ = np.unique(self.train_y_)
        self.train_median_ = np.nanmedian(self.train_X_[['fico', 'cltv', 'dti']], axis=0)
        self.ct_ = ColumnTransformer([('scaler', StandardScaler(), self.var_n_)])
        self.ohe_ = OneHotEncoder(drop='first', sparse=False)
        self.pe_ = PrepareEmbedding(cat_var_names=self.var_c3_)
        
        self.train_X_ = self.train_X_[['time'] + self.var_n_ + self.var_c2_ + self.var_c3_]
        self.train_X_ = self.train_X_.fillna(value={'fico': self.train_median_[0], 'cltv': self.train_median_[1], 'dti':self.train_median_[2]})
        self.train_X_[self.var_n_] = self.ct_.fit_transform(self.train_X_)
        self.train_X_[self.ohe_.get_feature_names_out()] = self.ohe_.fit_transform(self.train_X_[self.var_c2_])
        self.train_X_ = self.train_X_.drop(columns=self.var_c2_)
        if self.train_X_.shape[0] < 1e+5:
            self.train_X_, self.train_y_ = SMOTENC(categorical_features=self.train_X_.columns.isin(self.var_c3_ + list(self.ohe_.get_feature_names_out())), random_state=2022).fit_resample(self.train_X_, self.train_y_)
        else:
            self.train_X_, self.train_y_ = RandomUnderSampler(random_state=seed).fit_resample(self.train_X_, self.train_y_)
        self.train_X_[self.var_c3_] = pd.DataFrame(np.vstack(self.pe_.fit(data_fit=self.train_X_).transform()).T, columns=self.var_c3_, index=self.train_X_.index)
        self.var_n_c2_ = self.var_n_ + list(self.ohe_.get_feature_names_out())
        layer_size = np.ceil(self.layer_size_factor * len(self.var_n_c2_)).astype(int)
        output_size = np.ceil(self.output_size_factor * layer_size).astype(int)

        self.formula_ = {'logits':f"~ {'+'.join(self.var_n_c2_)} + spline(time, bs='bs', df={self.spline_num_knots + self.spline_degree}, degree={self.spline_degree}) + d_t(time) + d_n({', '.join(self.var_n_c2_)}) + {' + '.join(['d_c'+str(i)+'('+col+')' for i, col in enumerate(self.var_c3_)])}"}
        d_n = {
            'd_n':{
                'model': CustomeNNModule(
                    input_size=len(self.var_n_c2_),
                    num_hidden_layers=self.num_hidden_layers,
                    layer_size=layer_size,
                    activation=self.activation,
                    output_size=output_size
                    ),
                'output_shape': output_size
            },
            'd_t':{
                'model': CustomeNNModule(
                    input_size=1,
                    num_hidden_layers=self.num_hidden_layers,
                    layer_size=np.ceil(self.layer_size_factor * 2).astype(int),
                    activation=self.activation,
                    output_size=np.ceil(self.output_size_factor * self.layer_size_factor * 2).astype(int)
                    ),
                'output_shape': np.ceil(self.output_size_factor * self.layer_size_factor * 2).astype(int)
            }
        }
        d_c = {
            f'd_c{i}':{
                'model': CustomeNNModule(
                    input_size=self.pe_.get_num_embeddings()[col],
                    num_hidden_layers=self.num_hidden_layers,
                    layer_size=np.ceil(self.layer_size_factor * self.pe_.get_embedding_dim()[col]).astype(int),
                    activation=self.activation,
                    output_size=np.ceil(self.output_size_factor * self.layer_size_factor * self.pe_.get_embedding_dim()[col]).astype(int),
                    embedding=True,
                    embedding_dim=self.pe_.get_embedding_dim()[col]
                    ),
                'output_shape': np.ceil(self.output_size_factor * self.layer_size_factor * self.pe_.get_embedding_dim()[col]).astype(int)
                }
            for i, col in enumerate(self.var_c3_)
        }
        self.deep_models_dict_ = {**d_n, **d_c}
        self.train_parameters_ = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'optimizer_params': {'lr': self.learning_rate, 'momentum': 1 - self.momentum},
            'degrees_of_freedom': {'logits': self.spline_lambda_df_diff + self.spline_degree + self.spline_num_knots},
            'val_split': self.val_split,
            'early_stop_epochs': self.early_stop_epochs,
            'early_stop_epsilon': self.early_stop_epsilon,
            'dropout_rate': self.dropout_rate
        }

        self.sddr_ = Sddr(
            distribution=self.distribution,
            formulas=self.formula_,
            deep_models_dict=self.deep_models_dict_,
            train_parameters=self.train_parameters_,
            output_dir=self.output_dir)

        warnings.filterwarnings("ignore", category=UserWarning)
        self.sddr_.train(structured_data=self.train_X_, target=self.train_y_, plot=plot)

        return self

    def load(self, X, y, path_file, seed=None):

        """ Method docstring """

        self.train_X_ = deepcopy(X)
        self.train_y_ = deepcopy(y)
        if self.train_X_.shape[0] != self.train_y_.shape[0]:
            raise ValueError(f"X and y have different number of rows. X:{self.train_X_.shape}, y:{self.train_y_.shape}")
        self.var_n_ = ['fico', 'mi_pct', 'cnt_units', 'cltv', 'dti', 'current_upb', 'int_rt', 'orig_loan_term']
        self.var_c2_ = ['flag_fthb', 'cnt_borr', 'flag_sc', 'program_ind']
        self.var_c3_ = ['occpy_sts', 'channel', 'prop_type', 'loan_purpose', 'seller_name', 'servicer_name', 'pre_harp', 'cd_msa']
        self.classes_ = np.unique(self.train_y_)
        self.train_median_ = np.nanmedian(self.train_X_[['fico', 'cltv', 'dti']], axis=0)
        self.ct_ = ColumnTransformer([('scaler', StandardScaler(), self.var_n_)])
        self.ohe_ = OneHotEncoder(drop='first', sparse=False)
        self.pe_ = PrepareEmbedding(cat_var_names=self.var_c3_)
        
        self.train_X_ = self.train_X_[['time'] + self.var_n_ + self.var_c2_ + self.var_c3_]
        self.train_X_ = self.train_X_.fillna(value={'fico': self.train_median_[0], 'cltv': self.train_median_[1], 'dti':self.train_median_[2]})
        self.train_X_[self.var_n_] = self.ct_.fit_transform(self.train_X_)
        self.train_X_[self.ohe_.get_feature_names_out()] = self.ohe_.fit_transform(self.train_X_[self.var_c2_])
        self.train_X_ = self.train_X_.drop(columns=self.var_c2_)
        if self.train_X_.shape[0] < 1e+5:
            self.train_X_, self.train_y_ = SMOTENC(categorical_features=self.train_X_.columns.isin(self.var_c3_ + list(self.ohe_.get_feature_names_out())), random_state=2022).fit_resample(self.train_X_, self.train_y_)
        else:
            self.train_X_, self.train_y_ = RandomUnderSampler(random_state=seed).fit_resample(self.train_X_, self.train_y_)
        self.train_X_[self.var_c3_] = pd.DataFrame(np.vstack(self.pe_.fit(data_fit=self.train_X_).transform()).T, columns=self.var_c3_, index=self.train_X_.index)
        self.var_n_c2_ = self.var_n_ + list(self.ohe_.get_feature_names_out())
        layer_size = np.ceil(self.layer_size_factor * len(self.var_n_c2_)).astype(int)
        output_size = np.ceil(self.output_size_factor * layer_size).astype(int)

        self.formula_ = {'logits':f"~ {'+'.join(self.var_n_c2_)} + spline(time, bs='bs', df={self.spline_num_knots + self.spline_degree}, degree={self.spline_degree}) + d_t(time) + d_n({', '.join(self.var_n_c2_)}) + {' + '.join(['d_c'+str(i)+'('+col+')' for i, col in enumerate(self.var_c3_)])}"}
        d_n = {
            'd_n':{
                'model': CustomeNNModule(
                    input_size=len(self.var_n_c2_),
                    num_hidden_layers=self.num_hidden_layers,
                    layer_size=layer_size,
                    activation=self.activation,
                    output_size=output_size
                    ),
                'output_shape': output_size
            },
            'd_t':{
                'model': CustomeNNModule(
                    input_size=1,
                    num_hidden_layers=self.num_hidden_layers,
                    layer_size=np.ceil(self.layer_size_factor * 2).astype(int),
                    activation=self.activation,
                    output_size=np.ceil(self.output_size_factor * self.layer_size_factor * 2).astype(int)
                    ),
                'output_shape': np.ceil(self.output_size_factor * self.layer_size_factor * 2).astype(int)
            }
        }
        d_c = {
            f'd_c{i}':{
                'model': CustomeNNModule(
                    input_size=self.pe_.get_num_embeddings()[col],
                    num_hidden_layers=self.num_hidden_layers,
                    layer_size=np.ceil(self.layer_size_factor * self.pe_.get_embedding_dim()[col]).astype(int),
                    activation=self.activation,
                    output_size=np.ceil(self.output_size_factor * self.layer_size_factor * self.pe_.get_embedding_dim()[col]).astype(int),
                    embedding=True,
                    embedding_dim=self.pe_.get_embedding_dim()[col]
                    ),
                'output_shape': np.ceil(self.output_size_factor * self.layer_size_factor * self.pe_.get_embedding_dim()[col]).astype(int)
                }
            for i, col in enumerate(self.var_c3_)
        }
        self.deep_models_dict_ = {**d_n, **d_c}
        self.train_parameters_ = {
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'optimizer': self.optimizer,
            'optimizer_params': {'lr': self.learning_rate, 'momentum': 1 - self.momentum},
            'degrees_of_freedom': {'logits': self.spline_lambda_df_diff + self.spline_degree + self.spline_num_knots},
            'val_split': self.val_split,
            'early_stop_epochs': self.early_stop_epochs,
            'early_stop_epsilon': self.early_stop_epsilon,
            'dropout_rate': self.dropout_rate
        }

        self.sddr_ = Sddr(
            distribution=self.distribution,
            formulas=self.formula_,
            deep_models_dict=self.deep_models_dict_,
            train_parameters=self.train_parameters_,
            output_dir=self.output_dir)
        
        self.sddr_.load(path_file, self.train_X_)

    def predict(self, X):

        """ Method docstring """

        self.test_X_ = X
        self.test_X_ = self.test_X_[['time'] + self.var_n_ + self.var_c2_ + self.var_c3_]
        self.test_X_ = self.test_X_.fillna(value={'fico': self.train_median_[0], 'cltv': self.train_median_[1], 'dti':self.train_median_[2]})
        self.test_X_[self.var_n_] = self.ct_.transform(self.test_X_)
        self.test_X_[self.ohe_.get_feature_names_out()] = self.ohe_.transform(self.test_X_[self.var_c2_])
        self.test_X_ = self.test_X_.drop(columns=self.var_c2_)
        self.test_X_[self.var_c3_] = pd.DataFrame(np.vstack(self.pe_.transform(data_transform=self.test_X_)).T, columns=self.var_c3_, index=self.test_X_.index)
        max_time = self.train_X_['time'].max()
        self.test_X_['time'] = self.test_X_['time'].mask(self.test_X_['time'] > max_time, max_time)

        warnings.filterwarnings("ignore", category=UserWarning)
        self.distribution_layer_, self.partial_effect_ = self.sddr_.predict(self.test_X_, clipping=True, plot=False)

        return self.distribution_layer_.logits

    def score(self, X, y, sample_weight=None, subsample=True):

        """ Method docstring """

        self.test_X_ = deepcopy(X)
        self.test_y_ = deepcopy(y)
        if (self.test_X_.shape[0] > 1e+5 and subsample):
            indices = self.test_X_.index
            subsample_indices = np.random.choice(indices, 20000)
            self.test_X_ = self.test_X_.loc[subsample_indices,:]
            self.test_y_ = self.test_y_.loc[subsample_indices,:]
        self.pred_logits_ = self.predict(X=self.test_X_)
        self.pred_proba_ = expit(self.pred_logits_)

        return roc_auc_score(self.test_y_, self.pred_proba_)
