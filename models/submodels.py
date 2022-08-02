import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as torch_models

from models.hmr import hmr_backbone, hmr_regressor



# Get an image feature extraction model.
def get_feature_extraction_model(model='resnet50', pretrained=True):
    # Resnet-50 backbone
    if model == 'resnet50':
        _model = torch_models.__dict__[model](pretrained=pretrained)
        # Remove the last layer.
        return nn.Sequential(*list(_model.children())[:-1])
    
    # HMR backbone
    elif model == 'hmr_backbone':
        if pretrained:
            _model = hmr_backbone(pretrained=False)
            checkpoint = torch.load('data/hmr_model_checkpoint.pt', map_location=torch.device('cpu'))
            _model.load_state_dict(checkpoint['model'], strict=False)
            return _model
            
        # Call pretrained Resnet-50 if pretarined option is false.
        return hmr_backbone()
        
    # Custom model
    # model : log folder of the model
    else:
        _model = torch.load(os.path.join(model, 'model.pt'), map_location=torch.device('cpu'))
        if pretrained:
            _model.load_state_dict(torch.load(os.path.join(model, 'state_dict.pt'), map_location=torch.device('cpu')))
            
        return _model



# Get a position vector layer.
def get_position_vector_layer(model='hmr_regressor', pretrained=True, **kwargs):
    # HMR regressor
    if model == 'hmr_regressor':
        _model = hmr_regressor('data/smpl_mean_params.npz', **kwargs)
        
        if pretrained:
            checkpoint = torch.load('data/hmr_model_checkpoint.pt', map_location=torch.device('cpu'))
            _model.load_state_dict(checkpoint['model'], strict=False)
            
        return _model
    
    # Custom model
    # model : log folder of the model
    else:
        _model = torch.load(os.path.join(model, 'model.pt'), map_location=torch.device('cpu'))
        if pretrained:
            _model.load_state_dict(torch.load(os.path.join(model, 'state_dict.pt'), map_location=torch.device('cpu')))
        
        return _model



# Regression module
class ShapeExtractionLayer(nn.Module):
    def __init__(self, layer_info, extra_output_dim=None, dropout=0.1):
        """
            layer_info : number of neurons in layers from input to output
            extra_output_dim : extra output dimension, None for no extra output
        """
        super(ShapeExtractionLayer, self).__init__()
        i = layer_info[1]
        self.linear_layers = nn.ModuleList([nn.Linear(layer_info[0], i)])
        for j in layer_info[2:]:
            self.linear_layers.append(nn.Linear(i, j))
            i = j
        self.dropout_layer = nn.Dropout(p=dropout)
        self.activation_layer = nn.GELU()
        
        for layer in self.linear_layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        if extra_output_dim != None:
            self.extra_output_layer = nn.Linear(layer_info[-2], extra_output_dim)
            self.extra_output = True
            nn.init.xavier_uniform_(self.extra_output_layer.weight)
            nn.init.zeros_(self.extra_output_layer.bias)
        else:
            self.extra_output = False
            
    def forward(self, x):
        out = self.linear_layers[0](x)
        out = self.linear_layers[1](self.dropout_layer(self.activation_layer(out)))
        out = self.dropout_layer(out)
        if self.extra_output:
            return self.linear_layers[2](out), self.extra_output_layer(out)
        else:
            return self.linear_layers[2](out)
        
        
        
# Extra output (position vector) extraction layer
class ExtraOutputExtractionLayer(nn.Module):
    def __init__(self, layer_info, dropout=0.1):
        super(ExtraOutputExtractionLayer, self).__init__()
        """
            layer_info : number of neurons in layers from input to output
        """
        i = layer_info[1]
        self.linear_layers = nn.ModuleList([nn.Linear(layer_info[0], i), nn.Linear(i, layer_info[2])])
        self.extra_output_layer = nn.Linear(layer_info[2], layer_info[3])
        self.dropout_layer = nn.Dropout(p=dropout)
        self.activation_layer = nn.GELU()
        
        self.position_vector_len = layer_info[3]
            
    def forward(self, x):
        out = self.linear_layers[0](x)
        out = self.linear_layers[1](self.dropout_layer(self.activation_layer(out)))
        return self.extra_output_layer(self.dropout_layer(out))
    
    
# Extra output (position vector) estimator
class ExtraOutputExtractor(nn.Module):
    def __init__(self, backbone, extra_output_extraction_layer):
        super(ExtraOutputExtractor, self).__init__()
        self.backbone = backbone
        self.extra_output_extraction_layer = extra_output_extraction_layer
        self.position_vector_len = extra_output_extraction_layer.position_vector_len
            
    def forward(self, x):
        n_batch = x.shape[0]
        out = self.backbone(x).view(n_batch, -1)
        return self.extra_output_extraction_layer(out)
    
    
    
# Layer aggregating features using self-attention algorithm
class FeatureAggregationLayer(nn.Module):
    def __init__(self, d_embed, dropout=0.1):
        super(FeatureAggregationLayer, self).__init__()
        self.linear_layer = nn.Linear(d_embed, 1)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.softmax_layer = nn.Softmax(dim=-1)
        self.scale = 1 / np.sqrt(d_embed)
        
        nn.init.xavier_uniform_(self.linear_layer.weight)
        nn.init.zeros_(self.linear_layer.bias)
        
    def forward(self, x, masking_matrix=None):
        """
        x : (n_batch, max_seq_len, d_embed)
        masking_matrix : (n_batch, max_seq_len_in_batch), True values for padding values
        """
        scores = self.linear_layer(x).squeeze(dim=-1) * self.scale
        if masking_matrix != None:
            scores = scores + masking_matrix * (-1e9)
        weights = self.softmax_layer(scores)
        return self.dropout_layer(torch.matmul(weights.unsqueeze(1), x).squeeze(1))
