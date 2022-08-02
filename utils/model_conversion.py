import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import argparse

import numpy as np
import torch
import torch.nn as nn

from models.submodels import ExtraOutputExtractionLayer, ExtraOutputExtractor



def split_simple_model(model_dir, save_dir):
    # Load the simple model.
    model = torch.load(os.path.join(model_dir, 'model.pt'), map_location=torch.device('cpu'))
    model.load_state_dict(torch.load(os.path.join(model_dir, 'state_dict.pt'), map_location=torch.device('cpu')))
    
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    backbone_dir = os.path.join(save_dir, 'backbone/')
    position_vector_layer_dir = os.path.join(save_dir, 'position_vector_layer/')
    extra_output_extractor_dir = os.path.join(save_dir, 'extra_output_extractor/')
    if not os.path.exists(backbone_dir):
        os.mkdir(backbone_dir)
    if not os.path.exists(position_vector_layer_dir):
        os.mkdir(position_vector_layer_dir)
    if not os.path.exists(extra_output_extractor_dir):
        os.mkdir(extra_output_extractor_dir)
    
    # Save the backbone
    torch.save(model.backbone, os.path.join(backbone_dir, 'model.pt'))
    torch.save(model.backbone.state_dict(), os.path.join(backbone_dir, 'state_dict.pt'))
    
    # Save the position vector layer
    layer_info = model.shape_extraction_layer.linear_layers[0].weight.shape
    _layer_info = model.shape_extraction_layer.extra_output_layer.weight.shape
    layer_info = [layer_info[1], layer_info[0], _layer_info[1], _layer_info[0]]
    position_vector_layer = ExtraOutputExtractionLayer(layer_info)
    
    position_vector_layer.load_state_dict(model.shape_extraction_layer.state_dict(), strict=False)
    
    torch.save(position_vector_layer, os.path.join(position_vector_layer_dir, 'model.pt'))
    torch.save(position_vector_layer.state_dict(), os.path.join(position_vector_layer_dir, 'state_dict.pt'))
    
    extra_output_extractor = ExtraOutputExtractor(model.backbone, position_vector_layer)
    torch.save(extra_output_extractor, os.path.join(extra_output_extractor_dir, 'model.pt'))
    torch.save(extra_output_extractor.state_dict(), os.path.join(extra_output_extractor_dir, 'state_dict.pt'))
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True, type=str)
    parser.add_argument("--save_dir", required=True, type=str)
    
    options = parser.parse_args()
    split_simple_model(options.model_dir, options.save_dir)