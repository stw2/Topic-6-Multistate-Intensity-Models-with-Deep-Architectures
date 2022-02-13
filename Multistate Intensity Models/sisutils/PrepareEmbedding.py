import numpy as np
from sklearn.preprocessing import LabelEncoder

class PrepareEmbedding:

    def __init__(self, cat_var_names) -> None:
        if not all(isinstance(x, str) for x in cat_var_names):
            raise ValueError('Each elements in "cat_var_names" has to be a string.')
        self.cat_var_names = cat_var_names # list of names of categorical variables
        self.n_cat = len(cat_var_names) # number of categorical variables

    def fit(self, data_fit):
        # fit the training data into LabelEncoder()
        self.data_fit = data_fit
        self.maps = {cat: {levels: num for num, levels in enumerate(np.sort(self.data_fit[cat].unique()))} for cat in
                     self.cat_var_names} # maps between levels of each categorical variable and the assigned number
        self.label_encoder_list = self.data_fit[self.cat_var_names].apply(lambda x: LabelEncoder().fit(x))
        return self

    def transform(self, data_transform=None):
        # Let `data_transfrom` be None if we want to transform training data
        # If we want to transform the training data, let data_transfrom = test_dataframe
        if data_transform is None:
            labels_list = [self.label_encoder_list[i].transform(self.data_fit[self.cat_var_names[i]]) for i in range(self.n_cat)]
        else:
            for cat in self.cat_var_names:
                unseen_labels = np.setdiff1d(data_transform[cat].unique(), list(self.maps[cat].keys()))
                mask = np.isin(data_transform[cat], unseen_labels)
                data_transform[cat][mask] = 0
            labels_list = [self.label_encoder_list[i].transform(data_transform[self.cat_var_names[i]]) for i in range(self.n_cat)]
        return labels_list

    def get_num_embeddings(self, data_transform=None):
        # get the number of levels of each categorical variable, i.e., the number of embeddings
        if data_transform is None:
            return self.data_fit[self.cat_var_names].nunique()
        else:
            return data_transform[self.cat_var_names].nunique()

    def get_embedding_dim(self, data_transform=None):
        # get the dimension of embedding = min(50, #levels/2)
        num_emb = self.get_num_embeddings(data_transform)
        emb_dim = num_emb.apply(lambda x: min(50, x // 2))
        return emb_dim
