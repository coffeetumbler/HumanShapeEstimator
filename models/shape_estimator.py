import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import numpy as np
import torch
import torch.nn as nn

from models.transformer import get_transformer_encoder
from models.submodels import get_feature_extraction_model, get_position_vector_layer
from models.submodels import ShapeExtractionLayer, FeatureAggregationLayer

import utils.config as config



# Single-frame shape estimator
class SimpleShapeEstimator(nn.Module):
    def __init__(self, backbone, shape_extraction_layer):
        super(SimpleShapeEstimator, self).__init__()
        self.backbone = backbone
        self.shape_extraction_layer = shape_extraction_layer
        self.extra_output = shape_extraction_layer.extra_output
        
        with np.load(config.SMPL_MEAN_PARAMS_DIR) as smpl_mean_params:
            mean_shape = torch.from_numpy(smpl_mean_params['shape'].astype('float32'))
        self.register_buffer('mean_shape', mean_shape)
        
    def forward(self, x):
        n_batch = x.shape[0]
        
        out = self.backbone(x).view(n_batch, -1)
        out = self.shape_extraction_layer(out)
        
        if self.extra_output:
            return out[0] + self.mean_shape, out[1]
        else:
            return out + self.mean_shape
    
    
    
# Multi-frame shape estimator
class AggregationShapeEstimator(nn.Module):
    def __init__(self, backbone, linear_embedding_layer, transformer_encoder, aggregation_layer, shape_extraction_layer):
        super(AggregationShapeEstimator, self).__init__()
        self.backbone = backbone
        self.linear_embedding_layer = linear_embedding_layer
        self.transformer_encoder = transformer_encoder
        self.aggregation_layer = aggregation_layer
        self.shape_extraction_layer = shape_extraction_layer
        
        self.transformer_position_vector = transformer_encoder.return_position_vector
        self.linear_embedding = False if linear_embedding_layer == None else True
        
        with np.load(config.SMPL_MEAN_PARAMS_DIR) as smpl_mean_params:
            mean_shape = torch.from_numpy(smpl_mean_params['shape'].astype('float32'))
        self.register_buffer('mean_shape', mean_shape)
        
    def forward(self, x, n_batch, z=None):
        """
        <input>
        x : (n_img_in_batch, channel, width, height), n_img_in_batch = number of all images in a batch
        n_batch : number of sequences in a batch
        z : input for positional encoding layer, None for no positional encoding or the same input with x
        """
        n_img_in_batch = x.shape[0]
        device = x.device

        backbone_out = self.backbone(x).view(n_img_in_batch, -1).contiguous()
        _backbone_out = backbone_out
        if self.linear_embedding:
            backbone_out = self.linear_embedding_layer(_backbone_out)
        
        if self.transformer_position_vector:
            transformer_out, position_vector = self.transformer_encoder(x=backbone_out,
                                                                        n_batch=n_batch,
                                                                        masking_matrix=None,
                                                                        z=_backbone_out)
            aggregated_features = self.aggregation_layer(transformer_out)
            return self.shape_extraction_layer(aggregated_features) + self.mean_shape, position_vector
            
        else:
            transformer_out = self.transformer_encoder(x=backbone_out,
                                                       n_batch=n_batch,
                                                       masking_matrix=None,
                                                       z=z)
            aggregated_features = self.aggregation_layer(transformer_out)
            return self.shape_extraction_layer(aggregated_features) + self.mean_shape

    
    

# Get a single-frame model.
def get_simple_shape_estimator(backbone_options=config.BACKBONE_OPTIONS,
                               shape_extraction_options=config.SHAPE_EXTRACTION_OPTIONS):
    
    backbone = get_feature_extraction_model(**backbone_options)
    
    shape_extraction_layer_info = shape_extraction_options['layer_info'].copy()
    shape_extraction_layer_info[0] = config.FEATURE_DIM[backbone_options['model']]
    extra_output_dim = config.PARAMETER_DIM[shape_extraction_options['extra_output']][0]
    shape_extraction_layer = ShapeExtractionLayer(shape_extraction_layer_info,
                                                  extra_output_dim,
                                                  shape_extraction_options['dropout'])
    
    return SimpleShapeEstimator(backbone, shape_extraction_layer)
    

# Get a multi-frame model.
def get_aggregation_shape_estimator(backbone_options=config.BACKBONE_OPTIONS,
                                    position_vector_options=config.POSITION_VECTOR_OPTIONS,
                                    transformer_options=config.TRANSFORMER_OPTIONS.copy(),
                                    shape_extraction_options=config.SHAPE_EXTRACTION_OPTIONS):
    
    backbone = get_feature_extraction_model(**backbone_options)
    position_vector_layer = get_position_vector_layer(**position_vector_options) if transformer_options['positional_encoding'] != None else None
    
    if transformer_options['linear_embedding']:
        linear_embedding_layer = nn.Linear(2048, transformer_options['d_embed'])
        nn.init.xavier_uniform_(linear_embedding_layer.weight)
        nn.init.zeros_(linear_embedding_layer.bias)
    else:
        linear_embedding_layer = None

    del transformer_options['linear_embedding']
    transformer_encoder = get_transformer_encoder(position_vector_layer=position_vector_layer, **transformer_options)
    
    shape_extraction_layer_info = shape_extraction_options['layer_info'].copy()
    if transformer_options['positional_encoding'] in ['Concatenating', 'concatenating', 'cat']:
        shape_extraction_layer_info[0] = transformer_options['d_embed'] + position_vector_layer.position_vector_len
    else:
        shape_extraction_layer_info[0] = transformer_options['d_embed']
        
    shape_extraction_layer = ShapeExtractionLayer(layer_info=shape_extraction_layer_info,
                                                  extra_output_dim=None,
                                                  dropout=shape_extraction_options['dropout'])
    aggregation_layer = FeatureAggregationLayer(shape_extraction_layer_info[0], shape_extraction_options['dropout'])
    
    return AggregationShapeEstimator(backbone, linear_embedding_layer, transformer_encoder, aggregation_layer, shape_extraction_layer)
