""" Module docstring """

from torch import nn

class CustomeNNModule(nn.Module):

    """ Class docstring """

    def __init__(self, input_size, num_hidden_layers, layer_size, activation, output_size, embedding=False, embedding_dim=None) -> None:
        super(CustomeNNModule, self).__init__()

        self.embedding = embedding
        self.activation = activation
        if self.embedding:
            self.layers = nn.ModuleList([nn.Embedding(input_size, embedding_dim), nn.Flatten(start_dim=1)])
            self.layers.append(nn.Linear(embedding_dim, layer_size))
            self.layers.extend([nn.Linear(layer_size, layer_size) for i in range(num_hidden_layers-1)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, layer_size)])
            self.layers.extend([nn.Linear(layer_size, layer_size) for i in range(num_hidden_layers)])
        self.layers.append(nn.Linear(layer_size, output_size))

    def forward(self, x):

        """ Method docstring """

        if self.embedding:
            for i, layer in enumerate(self.layers):
                if i == 1:
                    x = layer(x)
                else:
                    x = self.activation(layer(x))
        else:
            for layer in self.layers:
                x = self.activation(layer(x))
        return x