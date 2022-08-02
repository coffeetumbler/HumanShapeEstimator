import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    
    # General options
    parser.add_argument("--model", required=True, type=str,
                        help="Name of model; simple/aggregation")
    parser.add_argument("--device", default='cuda', type=str,
                        help="cuda or cpu")
    parser.add_argument("--multi_gpu", default=False, action='store_true',
                        help="Use multi GPU")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--gpu_group", default=None, type=str,
                        help="Specified GPU group")
    parser.add_argument("--master_port", default='12355', type=str)
    
    parser.add_argument("--dataset", default='surreal', type=str,
                        help="Dataset for training; surreal/3dpw")
    parser.add_argument("--test_only", default=False, action='store_true',
                        help="Network test only")
    parser.add_argument("--continue_training", default=None, type=str,
                        help="Continue interrupted training; Directory for log files is needed.")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="Load the state dictionary checkpoint to model.")
    parser.add_argument("--comment", default=None, type=str,
                        help="Comments on training the model")
    
    parser.add_argument("--single_finetuning_options", default=False, action='store_true',
                        help="Recommended fine-tuning options for single-frame model; Input your pre-trained model checkpoint : your/model/checkpoint/state_dict.pt")
    parser.add_argument("--multi_pretraining_options", default=False, action='store_true',
                        help="Recommended pretraining options for multi-frame model")
    parser.add_argument("--multi_finetuning_options", default=False, action='store_true',
                        help="Recommended fine-tuning options for multi-frame model; Input your pre-trained model checkpoint : your/model/checkpoint/state_dict.pt")
    
    # Model options
    parser.add_argument("--embedding_dim", default=None, type=int)
    parser.add_argument("--dropout_ratio", default=None, type=float)
    parser.add_argument("--shape_parameter_dim", default=None, type=int)
    
    parser.add_argument("--custom_backbone", default=False, action='store_true',
                        help="Custom backbone options")
    parser.add_argument("--backbone_model", default='resnet50', type=str,
                        help="Backbone model; resnet50/hmr_backbone/")
    parser.add_argument("--backbone_untrained", dest='backbone_pretrained', default=True, action='store_false',
                        help="Backbone untrained option")
    
    parser.add_argument("--custom_position_vector", default=False, action='store_true',
                        help="Custom position vector options")
    parser.add_argument("--position_vector_model", default='hmr_regressor', type=str,
                        help="Position vector layer model; hmr_regressor")
    parser.add_argument("--position_vector_untrained", dest='position_vector_pretrained', default=True, action='store_false',
                        help="Position vector layer untrained option")
    parser.add_argument("--position_vector_global_rotation", default=False, action='store_true',
                        help="Global rotation in position vector")
    parser.add_argument("--position_vector_camera_params", default=False, action='store_true',
                        help="Camera parameters in position vector")
    
    parser.add_argument("--custom_transformer", default=False, action='store_true',
                        help="Custom transformer options")
    parser.add_argument("--transformer_positional_encoding", default=None, type=str,
                        help="Positional encoding type; Concatenating/None")
    parser.add_argument("--transformer_encoding_freezing", dest='transformer_encoding_trainable', default=True, action='store_false',
                        help="Positional encoding freezing(non-trainable) option")
    parser.add_argument("--transformer_return_position_vector", default=False, action='store_true',
                        help="Retruning position vector option")
    parser.add_argument("--transformer_remove_linear_embedding", dest='transformer_linear_embedding', default=True, action='store_false',
                        help="Linear embedding removal option before feeding data to transformer")
    parser.add_argument("--transformer_n_layer", default=3, type=int)
    parser.add_argument("--transformer_d_model", default=512, type=int)
    parser.add_argument("--transformer_n_head", default=8, type=int)
    parser.add_argument("--transformer_d_ff", default=2048, type=int)
    parser.add_argument("--transformer_max_seq_len", default=24, type=int)
    
    parser.add_argument("--custom_shape_extractor", default=False, action='store_true',
                        help="Custom shape extractor options")
    parser.add_argument("--shape_extractor_hidden_layer", default=2048, type=int,
                        help="Number of neurons in a hidden layer of shape extractor")
    parser.add_argument("--shape_extractor_extra_output", default=None, type=str,
                        help="Extra ouput for shape extractor; surreal_global_rotation/3dpw_global_rotation; default for no extra ouput")
    
    # Training options
    parser.add_argument("--num_epochs", default=50, type=int,
                        help="Total number of training epochs")
    parser.add_argument("--max_train_time", default=None, type=int,
                        help="Maximum training time; write in seconds")

    parser.add_argument("--augmentation", default='full', type=str,
                        help="Augmentation level for training dataset; full/noise/nonflip/none")
    parser.add_argument("--use_smpl_params", default=False, action='store_true',
                        help="Load SMPL parameters for each frame from dataset")
    
    parser.add_argument("--replacement", default=False, action='store_true',
                        help="Allowance to data replacement")
    parser.add_argument("--diff_n_seq_per_person", dest='equal_n_seq_per_person', default=True, action='store_false',
                        help="Inequal number of sequences per person")
    parser.add_argument("--n_seq_per_person", default=5, type=int,
                        help="Number of sequences per person")
    parser.add_argument("--no_shuffle_series", dest='shuffle_series', default=True, action='store_false',
                        help="No shuffle series when choosing frames for a person")
    parser.add_argument("--static_n_frames", default=False, action='store_true',
                        help="Fix the number of frames in a sequence")
    parser.add_argument("--frames_in_series", default=1, type=int,
                        help="(Mean) number of frames in a series(video) for training")
    parser.add_argument("--max_frames_in_seq", default=12, type=int,
                        help="maximum number of frames in a sequence")
    
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", "--learning_rate", default=1e-4, type=float)
    parser.add_argument("--warmup_lr", default=None, type=float,
                        help="Learning rate for warm-up epochs; default for no warm-up")
    parser.add_argument("--vertices_loss_weight", default=1., type=float,
                        help="Weight for the mesh vertices loss")
    parser.add_argument("--extra_loss_weight", default=1., type=float,
                        help="Weight for the extra output loss")
    parser.add_argument("--grad_clip_norm", default=1., type=float,
                        help="Maximum gradient norm in gradient clipping")
    parser.add_argument("--summary_steps", default=300, type=int,
                        help="Training summary frequency")
    
    return parser.parse_args()