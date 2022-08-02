import os, cv2, argparse
from tqdm import tqdm
import numpy as np
import torch

from utils.datasets import get_simple_dataset_loader, get_shape_focused_dataset_loader, get_test_dataset_loader
from models.smpl import SMPLStaticPose, blend_smpl_gender


# Test function
def test(args):
    device = torch.device('cuda:{}'.format(args.gpu_id)) if torch.cuda.is_available else torch.device('cpu')
    
    # SMPL models
    smpl = SMPLStaticPose().to(device)
    smpl_gender = []
    smpl_gender.append(SMPLStaticPose(gender=0).to(device))
    smpl_gender.append(SMPLStaticPose(gender=1).to(device))
    smpl.eval()
    smpl_gender[0].eval()
    smpl_gender[1].eval()
    
    # Load model and state dict.
    model = torch.load(args.model_dir, map_location=device)
    model.load_state_dict(torch.load(args.state_dict_dir, map_location='cpu'))
    model.eval()
    
    # Evaluation metric
    mPVE = lambda pred, gt: torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).sum()
    
    # Set sequence lengths for test.
    if args.model_type == 'simple':
        seq_lens = (1,)
    else:
        if args.sequence_length_range == None:
            seq_lens = range(1, 16) if args.sequence_length == None else (args.sequence_length,)
        else:
            seq_lens = range(*map(int, args.sequence_length_range.split(',')))
        
    # Losses and argument type
    losses = []
    arg_type = -1
    """
    arg_type = -1 : not identified yet
                0 : simple model yielding shape parameters only
                1 : simple model with extra output
                2 : hmr model
                3 : aggregation model yielding shape parameters only
                4 : aggregation model with position vector output
    """
        
    for i in seq_lens:
        # Get data loader.
        if args.dataset == 'surreal':
            if args.model_type == 'simple':
                data_loader = get_simple_dataset_loader(dataset=args.dataset,
                                                        set_type='test',
                                                        test_all=True,
                                                        batch_size=args.batch_size,
                                                        drop_last=False,
                                                        pin_memory=False)
                total_len = len(data_loader.sampler)
            else:
                data_loader = get_shape_focused_dataset_loader(dataset=args.dataset,
                                                               set_type='test',
                                                               protocol=True,
                                                               batch_size=args.batch_size,
                                                               mean_frames_in_seq=i,
                                                               drop_last=False,
                                                               pin_memory=False)
                total_len = len(data_loader.batch_sampler)
        else:
            data_loader = get_test_dataset_loader(dataset=args.dataset,
                                                  model_type=args.model_type,
                                                  protocol=True,
                                                  batch_size=args.batch_size,
                                                  mean_frames_in_seq=i,
                                                  drop_last=False,
                                                  pin_memory=False)
            total_len = len(data_loader.dataset) if args.model_type == 'simple' else len(data_loader.batch_sampler)
            
        # Evaluation.
        _mPVE = []
        for batch in tqdm(data_loader, total=total_len//args.batch_size):
            with torch.no_grad():
                images = batch['img'].to(device)
                gt_shape = batch['shape'].to(device)
                gt_vertices = blend_smpl_gender(smpl_gender, batch['gender'], gt_shape)
                
                # Record argument type once and get predictions according to argument type.
                if arg_type == 0:
                    pred_shape = model(images)
                elif arg_type == 1:
                    pred_shape, _ = model(images)
                elif arg_type == 2:
                    _, pred_shape, _ = model(images)
                elif arg_type == 3:
                    pred_shape = model(images, gt_shape.shape[0], images)
                elif arg_type == 4:
                    pred_shape, _ = model(images, gt_shape.shape[0], images)
                else:
                    num_inputs = model.forward.__code__.co_argcount
                    if num_inputs == 2:
                        outputs = model(images)
                        if isinstance(outputs, tuple):
                            if len(outputs) == 2:
                                pred_shape = outputs[0]
                                arg_type = 1
                            elif len(outputs) == 3:
                                pred_shape = outputs[1]
                                arg_type = 2
                        else:
                            pred_shape = outputs
                            arg_type = 0
                    elif num_inputs == 4:
                        outputs = model(images, gt_shape.shape[0], images)
                        if isinstance(outputs, tuple):
                            pred_shape = outputs[0]
                            arg_type = 4
                        else:
                            pred_shape = outputs
                            arg_type = 3
                            
                # Compute mesh vertices and errors.
                pred_vertices = smpl(pred_shape)
                _mPVE.append(mPVE(pred_vertices, gt_vertices).item())

        # Compute average error.
        result = np.sum(_mPVE) / total_len
        print('length : {0}, mPVE : {1}'.format(i, result))
        losses.append(result)
        
        del data_loader
        
    for i, loss in zip(seq_lens, losses):
        print('length : {0}, mPVE : {1}'.format(i, loss))
            
            
# Arguments for test
def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", required=True, type=str,
                        help="File name of the model")
    parser.add_argument("--model_type", required=True, type=str,
                        help="Type of the model; 'simple' for single-frame model and 'shape_focused' for multi-frame model")
    parser.add_argument("--state_dict_dir", required=True, type=str,
                        help="File name of the state dictionary")
    
    parser.add_argument("--dataset", default='3dpw', type=str,
                        help="test dataset; surreal/3dpw")
    parser.add_argument("--gpu_id", default=0, type=int,
                        help="GPU ID")
    parser.add_argument("--batch_size", default=4, type=int,
                        help="test dataset batch size")
    parser.add_argument("--sequence_length", default=None, type=int,
                        help="length of image sequence, default for 1 to 15, not used if model_type == simple")
    parser.add_argument("--sequence_length_range", default=None, type=str,
                        help="range of length of image sequence, first and last indices separated with comma(,) are required. (e.g. '1,16')")
    
    return parser.parse_args()
    
    
# Main function
if __name__ == '__main__':
    args = parse_args()
    test(args)
