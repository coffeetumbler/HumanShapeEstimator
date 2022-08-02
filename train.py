import os, cv2, copy, time
from tqdm import tqdm

import numpy as np

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from utils.datasets import get_simple_dataset_loader, get_shape_focused_dataset_loader, get_test_dataset_loader
import utils.config as config
from utils.options import parse_args

from models.shape_estimator import get_simple_shape_estimator, get_aggregation_shape_estimator
from models.smpl import SMPLStaticPose, blend_smpl_gender



# Base trainer
class BaseTrainer:
    def __init__(self, options):
        self.options = options
        if options.gpu_group != None:
            self.device = torch.device('cuda:' + options.gpu_group[0])
        else:
            self.device = torch.device('cuda') if options.device == 'cuda' and torch.cuda.is_available() else torch.device('cpu')
        
        self.set_type = 'test' if options.test_only else 'train'
        self.max_train_time = np.inf if options.max_train_time == None or options.max_train_time < 1\
                              else options.max_train_time
        
        """
        Required values
        self.model_name
        self.model
        self.smpl
        self.smpl_gender = [smpl_female, smpl_male]
        self.optimizer
        self.criterion
        self.train_data_loader
        self.train_data_len
        self.valid_data_loader
        self.trained_epoch if options.continue_training != None
        """
        
    # Training function
    def train(self):
        # Directory for log files
        log_dir = config.LOG_DIR + time.strftime('%y%m%d%H%M%S_', time.localtime(time.time())) + self.model_name if self.options.continue_training == None else self.options.continue_training
        summary_writer = SummaryWriter(log_dir)
        model_temp_dir = log_dir + '/temp_state'
        
        # Write comments for new training.
        if self.options.continue_training == None:
            if not os.path.exists(model_temp_dir):
                os.mkdir(model_temp_dir)

            if self.options.comment != None:
                with open(log_dir + '/comments.txt', 'w') as f:
                    split = self.options.comment.split(';')
                    for string in split:
                        f.write(string + '\n')

            torch.save(self.model, log_dir + '/model.pt')
            torch.save(self.options, log_dir + '/options.pt')
        
        # step infos
        summary_count = 1
        summary_logged_step = 0
        summary_steps = self.options.summary_steps
        
        resume_training = True
        end_time = self.max_train_time + time.time()
        
        # SMPL models for evalution
        self.smpl.eval()
        self.smpl_gender[0].eval()
        self.smpl_gender[1].eval()
        
        # initial epoch
        initial_epoch = 1 if self.options.continue_training == None else self.trained_epoch + 1
        
        # Start training.
        for epoch in tqdm(range(initial_epoch, self.options.num_epochs+1), desc='Training models'):
            # Train one epoch.
            with tqdm(total=self.train_data_len, desc='Epoch {}'.format(epoch)) as training_bar:
                self.model.train()
                for batch in self.train_data_loader:
                    if time.time() < end_time:
                        if summary_count < summary_steps:
                            loss = self.train_step(batch)
                            if loss != None:
                                summary_count += 1

                        else:
                            loss = self.train_step(batch)
                            if loss != None:
                                summary_logged_step += summary_steps
                                self.write_train_log(summary_writer, summary_logged_step, loss)
                                tqdm.write('Training summary logged; training loss : {}'.format(loss))
                                summary_count = 1
                            
                        training_bar.update(len(batch['gender']))

                    else:
                        tqdm.write('Timeout reached')
                        resume_training = False
                        break

            # Validate after one epoch training.
            if resume_training:
                self.model.eval()
                loss_str = self.validate(summary_writer, epoch)
                self.save_checkpoint(model_temp_dir, epoch)
                tqdm.write(loss_str)
            else:
                break
                    
        # Save checkpoint.
        self.save_checkpoint(log_dir)
                
    # Call interrupted model and state dict.
    def call_interrupted_model(self):
        self.model = torch.load(os.path.join(self.options.continue_training, 'model.pt'),
                                map_location=self.device)
        self.trained_epoch = 0
        for file_name in os.listdir(os.path.join(self.options.continue_training, 'temp_state/')):
            if 'state_dict_epoch' in file_name:
                self.trained_epoch += 1
        self.model.load_state_dict(torch.load(os.path.join(self.options.continue_training, 'temp_state/state_dict_epoch-{}.pt'.format(self.trained_epoch)),
                                              map_location=torch.device('cpu')))
        
    # Call specified model and state dict.
    def call_model(self, model_dir, state_dir, save_attr=None):
        model = torch.load(model_dir, map_location=self.device)
        model.load_state_dict(torch.load(state_dir, map_location=torch.device('cpu')), strict=False)
        if save_attr == None:
            return model
        else:
            setattr(self, save_attr, model)
            
    # Validation function
    def validate(self, summary_writer, epoch):
        valid_losses = []
        for batch in self.valid_data_loader:
            _loss = self.validate_step(batch)
            if _loss != None:
                valid_losses.append(_loss)
        loss = self.write_validate_log(summary_writer, epoch, valid_losses)
        loss_str = 'Validation implemented; '
        for key, value in loss.items():
            loss_str = loss_str + key + ' : {}, '.format(value)
        return loss_str[:-2]
        
    # Checkpoint saving function
    def save_checkpoint(self, log_dir, epoch=None):
        if epoch == None:
            torch.save(self.model.state_dict(), log_dir + '/state_dict.pt')
        else:
            torch.save(self.model.state_dict(), log_dir + '/state_dict_epoch-{}.pt'.format(epoch))
            
    # Functions needed to be defined in an inherited trainer
    def train_step(self, batch):
        raise NotImplementedError('"train_step" function is not provided.')
        
    def validate_step(self, batch):
        raise NotImplementedError('"validate_step" function is not provided.')
        
    def write_train_log(self, writer, step, loss):
        raise NotImplementedError('"write_train_log" function is not provided.')
        
    def write_validate_log(self, writer, epoch, losses):
        raise NotImplementedError('"write_trian_log" function is not provided.')
        
        
        
# Simple trainer for single-frame estimator
class SimpleTrainer(BaseTrainer):
    def __init__(self, options):
        super(SimpleTrainer, self).__init__(options)
        self.model_name = 'simple_shape_estimator'
        
        # Set training options.
        if options.continue_training != None:
            self.call_interrupted_model()
            
        else:
            backbone_options = config.BACKBONE_OPTIONS.copy()
            shape_extraction_options = copy.deepcopy(config.SHAPE_EXTRACTION_OPTIONS)

            embedding_dim = options.embedding_dim
            dropout_ratio = options.dropout_ratio
            shape_parameter_dim = options.shape_parameter_dim

            if options.custom_backbone:
                for key in backbone_options.keys():
                    backbone_options[key] = getattr(options, 'backbone_' + key)

            if options.custom_shape_extractor:
                shape_extraction_options['layer_info'][1] = options.shape_extractor_hidden_layer
                shape_extraction_options['layer_info'][2] = options.shape_extractor_hidden_layer
                if options.shape_extractor_extra_output != None:
                    shape_extraction_options['extra_output'] = options.shape_extractor_extra_output

            config.update_config(dropout_ratio=options.dropout_ratio,
                                 shape_parameter_dim=options.shape_parameter_dim,
                                 shape_extraction_options=shape_extraction_options)
        
            self.model = get_simple_shape_estimator(backbone_options, shape_extraction_options).to(self.device)
            
            if options.checkpoint != None:
                self.model.load_state_dict(torch.load(options.checkpoint, map_location=torch.device('cpu')), strict=False)
                
        # Extra output (global rotation) settings
        if options.shape_extractor_extra_output != None:
            self.extra_output = True
            self.extra_output_index = range(*(config.PARAMETER_DIM[options.shape_extractor_extra_output][1]))
            self.nonflip_augmentation = True if options.augmentation == 'nonflip' else False
        else:
            self.extra_output = False
        
        # SMPL models
        self.smpl = SMPLStaticPose().to(self.device)
        self.smpl_gender = []
        self.smpl_gender.append(SMPLStaticPose(gender=0).to(self.device))
        self.smpl_gender.append(SMPLStaticPose(gender=1).to(self.device))
        
        # Optimizer and main criterions
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=options.lr, weight_decay=1e-4)
        
        self.criterion_mesh = nn.L1Loss().to(self.device)
        self.mPVE = lambda pred, gt : torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).sum()
        
        # Criterion for extra output
        if self.extra_output:
            self.criterion_extra = nn.MSELoss().to(self.device)
        
        # Get dataset loaders.
        if options.dataset == 'surreal':
            self.train_data_loader = get_simple_dataset_loader(dataset=options.dataset,
                                                               set_type=self.set_type,
                                                               augmentation=options.augmentation,
                                                               use_smpl_params=options.use_smpl_params,
                                                               batch_size=options.batch_size,
                                                               frames_in_series=options.frames_in_series,
                                                               replacement=False,
                                                               pin_memory=False,
                                                               drop_last=True)

            self.train_data_len = len(self.train_data_loader.dataset) * options.frames_in_series

            self.valid_data_loader = get_simple_dataset_loader(dataset=options.dataset,
                                                               set_type='valid',
                                                               use_smpl_params=options.use_smpl_params,
                                                               batch_size=options.batch_size,
                                                               frames_in_series=options.frames_in_series,
                                                               replacement=False,
                                                               pin_memory=False,
                                                               drop_last=False)

            self.valid_data_len = len(self.valid_data_loader.dataset) * options.frames_in_series
            
        else:
            self.train_data_loader = get_test_dataset_loader(dataset=options.dataset,
                                                             set_type=self.set_type,
                                                             batch_size=options.batch_size,
                                                             augmentation=options.augmentation,
                                                             use_smpl_params=options.use_smpl_params,
                                                             pin_memory=False,
                                                             drop_last=True)
            self.train_data_len = len(self.train_data_loader.dataset)
            
            self.valid_data_loader = get_test_dataset_loader(dataset=options.dataset,
                                                             set_type='test',
                                                             batch_size=options.batch_size,
                                                             use_smpl_params=options.use_smpl_params,
                                                             pin_memory=False,
                                                             drop_last=False)
            self.valid_data_len = len(self.valid_data_loader.dataset)
           
        
    # One step function for training
    def train_step(self, batch):
        # Set inputs and ground truth values.
        images = batch['img'].to(self.device)
        gt_shape = batch['shape'].to(self.device)
        
        with torch.no_grad():
            gt_vertices = blend_smpl_gender(self.smpl_gender, batch['gender'], gt_shape)
            
        if self.extra_output:  # Extra output prediction
            gt_extra = batch['smpl'][:, self.extra_output_index].to(self.device)
            if self.nonflip_augmentation:
                gt_extra = torch.matmul(batch['rotation'].to(self.device), gt_extra.view(-1, 3, 3)).view(-1, 9)
            pred_shape, pred_extra = self.model(images)
        else:
            pred_shape = self.model(images)
            
        pred_vertices = self.smpl(pred_shape)  # Vertices prediction
        
        # Compute loss.
        loss = self.options.vertices_loss_weight * self.criterion_mesh(pred_vertices, gt_vertices)
        
        if self.extra_output:
            loss += self.options.extra_loss_weight * self.criterion_extra(pred_extra, gt_extra)
        
        # Backwarding
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        nn.utils.clip_grad_norm_(self.model.parameters(), self.options.grad_clip_norm)
            
        self.optimizer.step()
        
        return loss.item()
    
    
    # One step function for validation
    def validate_step(self, batch):
        with torch.no_grad():
            # Set inputs and ground truth values.
            images = batch['img'].to(self.device)
            gt_shape = batch['shape'].to(self.device)
            gt_vertices = blend_smpl_gender(self.smpl_gender, batch['gender'], gt_shape)
            
            n_batch = gt_shape.shape[0]
            
            if self.extra_output:  # Extra output prediction
                # No augmentation in validation steps.
                gt_extra = batch['smpl'][:, self.extra_output_index].to(self.device)
                pred_shape, pred_extra = self.model(images)
            else:
                pred_shape = self.model(images)
            
            pred_vertices = self.smpl(pred_shape)  # Vertices prediction
            
            # Record losses.
            losses = {}
            losses['mPVE'] = self.mPVE(pred_vertices, gt_vertices).item()
            if self.extra_output:
                losses['extra_output'] = self.criterion_extra(pred_extra, gt_extra).item() * n_batch
            
        return losses
        
        
    # Write logs.
    def write_train_log(self, writer, step, loss):
        writer.add_scalar('Loss/Train', loss, step)
        
    def write_validate_log(self, writer, epoch, losses):
        display = {}
        for key in losses[0].keys():
            loss = np.sum([l[key] for l in losses]) / self.valid_data_len
            writer.add_scalar('Loss/Valid : ' + key, loss, epoch)
            display[key] = loss
        return display
    
    
    
    
# Aggregated model trainer for multi-frame estimator
class AggregationModelTrainer(BaseTrainer):
    def __init__(self, options):
        super(AggregationModelTrainer, self).__init__(options)
        self.model_name = 'aggregation_shape_estimator'
        
        # Set training options.
        if self.options.continue_training != None:
            self.call_interrupted_model()
            
        else:
            backbone_options = config.BACKBONE_OPTIONS.copy()
            position_vector_options = config.POSITION_VECTOR_OPTIONS.copy()
            transformer_options = config.TRANSFORMER_OPTIONS.copy()
            shape_extraction_options = copy.deepcopy(config.SHAPE_EXTRACTION_OPTIONS)

            if options.custom_backbone:
                for key in backbone_options.keys():
                    backbone_options[key] = getattr(options, 'backbone_' + key)

            custom_position_vector = True
            if options.custom_transformer:
                for key in transformer_options.keys():
                    if key != 'd_embed' and key != 'dropout':
                        transformer_options[key] = getattr(options, 'transformer_' + key)
                if options.transformer_positional_encoding not in ['Concatenating', 'concatenating', 'cat']:
                    custom_position_vector = False

            if options.custom_position_vector and custom_position_vector:
                for key in position_vector_options.keys():
                    if key != 'd_embed' and key != 'dropout':
                        position_vector_options[key] = getattr(options, 'position_vector_' + key)

            if options.custom_shape_extractor:
                shape_extraction_options['layer_info'][1] = options.shape_extractor_hidden_layer
                shape_extraction_options['layer_info'][2] = options.shape_extractor_hidden_layer

            config.update_config(embedding_dim=options.embedding_dim,
                                 dropout_ratio=options.dropout_ratio,
                                 shape_parameter_dim=options.shape_parameter_dim,
                                 position_vector_options=position_vector_options,
                                 transformer_options=transformer_options,
                                 shape_extraction_options=shape_extraction_options)

            self.model = get_aggregation_shape_estimator(backbone_options,
                                                         position_vector_options,
                                                         transformer_options,
                                                         shape_extraction_options).to(self.device)
            
            if options.checkpoint != None:
                self.model.load_state_dict(torch.load(options.checkpoint, map_location=torch.device('cpu')), strict=False)
                
        # Trainable position vector (global rotation) settings
        self.transformer_encoding_trainable = options.transformer_encoding_trainable
        self.return_position_vector = options.transformer_return_position_vector
            
        # SMPL models
        self.smpl = SMPLStaticPose().to(self.device)
        self.smpl_gender = []
        self.smpl_gender.append(SMPLStaticPose(gender=0).to(self.device))
        self.smpl_gender.append(SMPLStaticPose(gender=1).to(self.device))
        
        # Optimizer and main criterions
        self.optimizer = torch.optim.AdamW(params=self.model.parameters(), lr=self.options.lr, weight_decay=1e-4)
        
        self.criterion_mesh = nn.L1Loss().to(self.device)
        self.mPVE = lambda pred, gt : torch.sqrt(((pred - gt) ** 2).sum(dim=-1)).mean(dim=-1).sum()
        
        # Criterion for position vector
        if self.return_position_vector:
            self.criterion_extra = nn.MSELoss().to(self.device)
            self.extra_output_index = range(*(config.PARAMETER_DIM[options.dataset+'_global_rotation'][1]))
            self.nonflip_augmentation = True if options.augmentation == 'nonflip' else False
        
        # Get dataset loaders.
        if options.dataset == 'surreal':
            self.train_data_loader = get_shape_focused_dataset_loader(dataset=options.dataset,
                                                                      set_type=self.set_type,
                                                                      augmentation=options.augmentation,
                                                                      use_smpl_params=options.use_smpl_params,
                                                                      batch_size=options.batch_size,
                                                                      replacement=options.replacement,
                                                                      equal_n_seq_per_person=options.equal_n_seq_per_person,
                                                                      n_seq_per_person=options.n_seq_per_person,
                                                                      shuffle_series=options.shuffle_series,
                                                                      static_n_frames=options.static_n_frames,
                                                                      mean_frames_in_seq=options.frames_in_series,
                                                                      max_frames_in_seq=options.max_frames_in_seq,
                                                                      drop_last=True,
                                                                      pin_memory=False)

            self.train_data_len = len(self.train_data_loader.batch_sampler)

            self.valid_data_loader = get_shape_focused_dataset_loader(dataset=options.dataset,
                                                                      set_type='valid',
                                                                      use_smpl_params=options.use_smpl_params,
                                                                      batch_size=options.batch_size,
                                                                      equal_n_seq_per_person=False,
                                                                      shuffle_series=False,
                                                                      static_n_frames=False,
                                                                      mean_frames_in_seq=5,
                                                                      max_frames_in_seq=options.max_frames_in_seq,
                                                                      drop_last=True,
                                                                      pin_memory=False)

            self.valid_data_len = len(self.valid_data_loader.batch_sampler)
            
        else:
            self.train_data_loader = get_test_dataset_loader(dataset=options.dataset,
                                                             model_type=options.model,
                                                             set_type=self.set_type,
                                                             batch_size=options.batch_size,
                                                             augmentation=options.augmentation,
                                                             use_smpl_params=options.use_smpl_params,
                                                             n_seq_per_person=options.n_seq_per_person,
                                                             static_n_frames=options.static_n_frames,
                                                             mean_frames_in_seq=options.frames_in_series,
                                                             max_frames_in_seq=options.max_frames_in_seq,
                                                             pin_memory=False,
                                                             drop_last=True)
            self.train_data_len = len(self.train_data_loader.batch_sampler)
            
            self.valid_data_loader = get_test_dataset_loader(dataset=options.dataset,
                                                             model_type=options.model,
                                                             set_type='test',
                                                             batch_size=options.batch_size,
                                                             use_smpl_params=options.use_smpl_params,
                                                             n_seq_per_person=10,
                                                             static_n_frames=False,
                                                             mean_frames_in_seq=5,
                                                             max_frames_in_seq=options.max_frames_in_seq,
                                                             pin_memory=False,
                                                             drop_last=False)
            self.valid_data_len = len(self.valid_data_loader.batch_sampler)
           
        
    # One step function for training
    def train_step(self, batch):
        # Set inputs and ground truth values.
        images = batch['img'].to(self.device)
        gt_shape = batch['shape'].to(self.device)
        
        with torch.no_grad():
            gt_vertices = blend_smpl_gender(self.smpl_gender, batch['gender'], gt_shape)
        
        batch_factor = gt_shape.shape[0] / float(self.options.batch_size)

        # Get predictions.
        try:
            # Shape and position vector prediction
            if self.return_position_vector:
                pred_shape, pred_pos = self.model(images, gt_shape.shape[0])
                gt_pos = batch['smpl'][:, self.extra_output_index].to(self.device)
                if self.nonflip_augmentation:
                    gt_pos = torch.matmul(batch['rotation'].to(self.device), gt_pos.view(-1, 3, 3)).view(-1, 9)
            # Shape prediction only
            else:
                pred_shape = self.model(images, gt_shape.shape[0]) if self.transformer_encoding_trainable\
                             else self.model(images, gt_shape.shape[0], images)
            pred_vertices = self.smpl(pred_shape)

            # Compute loss.
            loss = self.options.vertices_loss_weight * self.criterion_mesh(pred_vertices, gt_vertices)
            
            if self.return_position_vector:
                loss += self.options.extra_loss_weight * self.criterion_extra(pred_pos, gt_pos)

            loss *= batch_factor
            
            # Backwarding
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(self.model.parameters(), self.options.grad_clip_norm)
            
            self.optimizer.step()
        except RuntimeError as e:
            print(' total frames :', images.shape[0])
            raise e
        
        return loss
    
    
    # One step function for validation
    def validate_step(self, batch):
        with torch.no_grad():
            # Set inputs and ground truth values.
            images = batch['img'].to(self.device)
            gt_shape = batch['shape'].to(self.device)
            gt_vertices = blend_smpl_gender(self.smpl_gender, batch['gender'], gt_shape)
            
            n_batch = gt_shape.shape[0]
            
            # Shape and position vector prediction
            if self.return_position_vector:
                pred_shape, pred_pos = self.model(images, n_batch)
                gt_pos = batch['smpl'][:, self.extra_output_index].to(self.device)
            # Shape prediction only
            else:
                pred_shape = self.model(images, n_batch) if self.transformer_encoding_trainable\
                             else self.model(images, n_batch, images)
            pred_vertices = self.smpl(pred_shape)
            
            # Record losses.
            losses = {}
            losses['mPVE'] = self.mPVE(pred_vertices, gt_vertices).item()
            if self.return_position_vector:
                losses['position_vector'] = self.criterion_extra(pred_pos, gt_pos).item() * n_batch
            
        return losses
        
        
    # Write logs.
    def write_train_log(self, writer, step, loss):
        writer.add_scalar('Loss/Train', loss, step)
        
    def write_validate_log(self, writer, epoch, losses):
        display = {}
        for key in losses[0].keys():
            loss = np.sum([l[key] for l in losses]) / self.valid_data_len
            writer.add_scalar('Loss/Valid : ' + key, loss, epoch)
            display[key] = loss
        return display
        
        
        
# Main function
if __name__ == "__main__":
    options = parse_args()
    if options.single_finetuning_options:
        checkpoint = options.checkpoint
        options = torch.load(config.DEFAULT_OPTIONS['single_fine-tune'])
        options.checkpoint = checkpoint
    elif options.multi_pretraining_options:
        options = torch.load(config.DEFAULT_OPTIONS['multi_training'])
    elif options.multi_finetuning_options:
        checkpoint = options.checkpoint
        options = torch.load(config.DEFAULT_OPTIONS['multi_fine-tune'])
        options.checkpoint = checkpoint
    elif options.continue_training != None:
        log_dir = options.continue_training
        options = torch.load(os.path.join(log_dir, 'options.pt'))
        options.continue_training = log_dir
    
    if options.model == 'simple':
        trainer = SimpleTrainer(options)
    elif options.model == 'aggregation':
        trainer = AggregationModelTrainer(options)
    trainer.train()