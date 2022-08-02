import os, cv2, argparse
import numpy as np
import torch

from utils.image_processing import process_image, crop
import utils.config as config
from utils.renderer import Renderer

from models.smpl import SMPLStaticPose, blend_smpl_gender



# Demo function
def demo(args):
    device = torch.device(args.device) if torch.cuda.is_available() else torch.device('cpu')  # Set device.

    # Get an image sequence info from 3DPW test protocol.
    with np.load(config.TEST_PROTOCOL_DIR['3dpw']) as fixed_indices:
        fixed_index = fixed_indices[str(args.n_frames)][args.test_index]
    
    # Get 3DPW test set info.
    with np.load(config.PW3D_DATASET_DIR['test'] + config.DATASET_INFO['simple'], allow_pickle=True) as data_infos:
        shape = data_infos['shape']
        imgname = data_infos['imgname']
        center = data_infos['center']
        scale = data_infos['scale']
        
    if not os.path.exists('examples/demo/'):
        if not os.path.exists('examples/'):
            os.mkdir('examples/')
        os.mkdir('examples/demo/')
        
    folder = 'examples/demo/3DPW_test_protocol_length_{0}_index_{1}/'.format(args.n_frames, args.test_index)
    if not os.path.exists(folder):
        os.mkdir(folder)
        
    # Save cropped images for input sequence.
    for i, _index in enumerate(fixed_index):
        _scale = scale[_index].copy()
        _center = center[_index].copy()
        frame_name = os.path.join(config.PW3D_DATASET_DIR['base'], imgname[_index])
        img = cv2.imread(frame_name).copy().astype(np.float32)
        img = crop(img, _center, 1.12*_scale, config.IMG_RES)
        cv2.imwrite(folder + 'cropped_img_{}.jpg'.format(i), img)
        
    # SMPL models
    smpl = SMPLStaticPose().to(device)
    smpl_gender = SMPLStaticPose(gender=1).to(device)  # Male only in 3DPW test set

    smpl.eval()
    smpl_gender.eval()
        
    renderer = Renderer(focal_length=2000, img_res=480)  # Renderer
    
    # Call the trained models.
    model_single = torch.load('logs/single-frame_estimator_fine-tuned/model.pt', map_location=device)
    model_single.load_state_dict(torch.load('logs/single-frame_estimator_fine-tuned/state_dict.pt', map_location='cpu'))
    model_single.eval()
    
    model_multi = torch.load('logs/multi-frame_estimator_fine-tuned/model.pt', map_location=device)
    model_multi.load_state_dict(torch.load('logs/multi-frame_estimator_fine-tuned/state_dict.pt', map_location='cpu'))
    model_multi.eval()
    
    mPVE = lambda pred, gt: torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1)  # Define MPVE.
    
    # Normalize images.
    norm_imgs = []
    for name in sorted(os.listdir(folder)):
        if '.jpg' in name:
            _, norm_img_model = process_image(folder+name)
            norm_imgs.append(norm_img_model)
            
    ftxt = open(folder+'mpve.txt', 'w')  # Evaluation records
    
    with torch.no_grad():
        # Get ground truth shape and draw the mesh.
        images = torch.cat(norm_imgs, dim=0).to(device)
        gt_shape = torch.Tensor([shape[fixed_index[0]].copy()]).to(device)
        gt_vertices = smpl_gender(gt_shape)
        img_mesh = renderer(gt_vertices[0].cpu(), [0., 0., 9.920635], color='blue', image=np.array([[[0,0,0,1]]]))
        cv2.imwrite(folder+'mesh_gt.png', 255 * img_mesh)

        # Estimate the common shape using multi-frame model and draw the mesh.
        pred_shape = model_multi(images, 1, images)
        pred_vertices = smpl(pred_shape)
        img_mesh = renderer(pred_vertices[0].cpu(), [0., 0., 9.920635], color='red', image=np.array([[[0,0,0,1]]]))
        cv2.imwrite(folder+'mesh_multi.png', 255 * img_mesh)

        # Compute MPVE.
        mpve = mPVE(gt_vertices, pred_vertices)
        ftxt.write('Multi-frame model MPVE : {}\n'.format(mpve[0]))
        
        # Estimate the shape of each frame using single-frame model and draw the meshes.
        pred_shape = model_single(images)
        pred_vertices = smpl(pred_shape)
        for i in range(args.n_frames):
            img_mesh = renderer(pred_vertices[i].cpu(), [0., 0., 9.920635], color='red', image=np.array([[[0,0,0,1]]]))
            cv2.imwrite(folder+'mesh_single_{}.png'.format(i), 255 * img_mesh)

        # Compute MPVEs.
        mpve = mPVE(gt_vertices.expand(args.n_frames, -1, -1), pred_vertices)
        ftxt.write('Single-frame model MPVE :\n')
        for _mpve in mpve:
            ftxt.write('{},\n'.format(_mpve))
            
    ftxt.close()



# Arguments for demo
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--device", default='cuda', type=str)
    parser.add_argument("--n_frames", default=6, type=int,
                        help="Number of frames in an input sequence")
    parser.add_argument("--test_index", default=2505, type=int,
                        help="Index of the input sequence in 3DPW test protocol; 0 to 3699")
    
    return parser.parse_args()
    
    
# Main function
if __name__ == '__main__':
    args = parse_args()
    demo(args)