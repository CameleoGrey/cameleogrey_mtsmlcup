import torch
import torch.nn as nn

class MLP_Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layer_dim, hidden_layers_num=3, dropout_rate=0.05):
        super( MLP_Network, self).__init__()

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        
        if isinstance( hidden_layer_dim, tuple ):
            hidden_layer_dim = hidden_layer_dim[0]
        
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_dim),
            nn.BatchNorm1d(hidden_layer_dim),
            nn.ReLU(inplace=True)
        )
        self.input_layer.apply(init_weights)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_layer_dim, hidden_layer_dim),
            nn.BatchNorm1d(hidden_layer_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_layer_dim, output_dim),
        )
        self.output_layer.apply(init_weights)
        

        self.middle_layers = []
        for i in range( hidden_layers_num ):
            hidden_layer = nn.Sequential(
                nn.Linear(hidden_layer_dim, hidden_layer_dim),
                nn.BatchNorm1d(hidden_layer_dim),
                nn.Dropout(p=dropout_rate),
                nn.ReLU(inplace=True),
            )
            hidden_layer.apply(init_weights)
            self.middle_layers.append( hidden_layer )
        self.middle_layers = nn.Sequential( * self.middle_layers )

        pass

    def forward(self, x):
        x = self.input_layer(x)
        x = self.middle_layers(x)
        x = self.output_layer(x)
        return x