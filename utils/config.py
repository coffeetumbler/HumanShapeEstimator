import os

PROJECT_DIR = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
LOG_DIR = PROJECT_DIR + '/logs/'
BASE_DATASET_DIR = ''  # Write your/dataset/directory/

SMPL_MEAN_PARAMS_DIR = PROJECT_DIR + '/data/smpl_mean_params.npz'
J_REGRESSOR_DIR = {'neutral' : PROJECT_DIR + '/data/J_regressor_neutral.pt',
                   'male' : PROJECT_DIR + '/data/J_regressor_male.pt',
                   'female' : PROJECT_DIR + '/data/J_regressor_female.pt'}
SMPL_FACES_DIR = {'neutral' : PROJECT_DIR + '/data/smpl_faces_neutral.npy',
                  'male' : PROJECT_DIR + '/data/smpl_faces_male.npy',
                  'female' : PROJECT_DIR + '/data/smpl_faces_female.npy'}

TEST_PROTOCOL_DIR = {'surreal' : PROJECT_DIR + '/data/protocol/test_protocol_surreal.npz',
                     '3dpw' : PROJECT_DIR + '/data/protocol/test_protocol_3dpw.npz'}
DEFAULT_OPTIONS = {'single_fine-tune' : PROJECT_DIR + '/data/options/single_fine-tuning_default_options.pt',
                   'multi_training' : PROJECT_DIR + '/data/options/multi_training_default_options.pt',
                   'multi_fine-tune' : PROJECT_DIR + '/data/options/multi_fine-tuning_default_options.pt'}



FOCAL_LENGTH = 1000

# BGR order
IMG_NORM_MEAN = [0.406, 0.456, 0.485]
IMG_NORM_STD = [0.225, 0.224, 0.229]

IMG_RES = [224, 224]
IMG_CENTER = [112, 112]
IMG_NOISE_RANGE = [1 - 0.4, 1 + 0.4]
IMG_ROTATION_FACTOR = 30
IMG_SCALE_FACTOR = 0.25
IMG_DEFAULT_SCALE = 224 / 200.



EMBEDDING_DIM = 512
DROPOUT_RATIO = 0.1
SHAPE_PARAMETER_DIM = 10


FEATURE_DIM = {'resnet50' : 2048,
               'hmr_backbone' : 2048
              }

PARAMETER_DIM = {None : (None, None),
                 'None' : (None, None),
                 'surreal_global_rotation' : (9, (8,17)),
                 '3dpw_global_rotation' : (9, (0,9))
                }


BACKBONE_OPTIONS = {'model' : 'resnet50',
                    'pretrained' : True
                   }


POSITION_VECTOR_OPTIONS = {'model' : 'hmr_regressor',
                           'pretrained' : True,
                           'd_embed' : 2048,  # input feature dimension
                           'global_rotation' : False,
                           'camera_params' : False,
                           'dropout' : DROPOUT_RATIO
                          }


TRANSFORMER_OPTIONS = {'d_embed' : EMBEDDING_DIM,
                       'positional_encoding' : None,
                       'encoding_trainable' : True,
                       'return_position_vector' : False,
                       'linear_embedding' : True,
                       'n_layer' : 3,
                       'd_model' : 512,
                       'n_head' : 8,
                       'd_ff' : 2048,
                       'max_seq_len' : 24,
                       'dropout' : DROPOUT_RATIO
                      }

    
SHAPE_EXTRACTION_OPTIONS = {'layer_info' : [2048, 2048, 2048, SHAPE_PARAMETER_DIM],
                            'extra_output' : None,
                            'dropout' : DROPOUT_RATIO
                           }



SURREAL_DATASET_DIR = {'train' : BASE_DATASET_DIR + 'SURREAL/data/cmu/train/',
                       'test' : BASE_DATASET_DIR + 'SURREAL/data/cmu/test/',
                       'valid' : BASE_DATASET_DIR + 'SURREAL/data/cmu/val/'
                      }

PW3D_DATASET_DIR = {'base' : BASE_DATASET_DIR + '3DPW/',
                    'train' : BASE_DATASET_DIR + '3DPW/infos/train/',
                    'test' : BASE_DATASET_DIR + '3DPW/infos/test/',
                    'valid' : BASE_DATASET_DIR + '3DPW/infos/val/'
                   }

DATASET_DIR = {'surreal' : SURREAL_DATASET_DIR,
               '3dpw' : PW3D_DATASET_DIR
              }

DATASET_INFO = {'simple' : 'data_infos/simple_info.npz',
                'shape_focused' : 'data_infos/shape_focused_info.npz'
               }



def update_config(embedding_dim=None,
                  dropout_ratio=None,
                  shape_parameter_dim=None,
                  position_vector_options=None,
                  transformer_options=None,
                  shape_extraction_options=None):
    
    if embedding_dim != None:
        if transformer_options != None:
            transformer_options['d_embed'] = embedding_dim
    
    if dropout_ratio != None:
        if position_vector_options != None:
            position_vector_options['dropout'] = dropout_ratio
        if transformer_options != None:
            transformer_options['dropout'] = dropout_ratio
        if shape_extraction_options != None:
            shape_extraction_options['dropout'] = dropout_ratio
            
    if shape_parameter_dim != None:
        if shape_extraction_options != None:
            shape_extraction_options['layer_info'][3] = shape_parameter_dim