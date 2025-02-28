config = {
 'expname': 'zjuhuman_explicit',
 'logdir': './logs/syntheticdynamic',
 'device': 'cuda:0',

 'data_downsample': 1.0,
 'data_dirs': ['data/dnerf/data/zjuhuman'],
 'contract': False,
 'ndc': False,
 'isg': False,
 'isg_step': -1,
 'ist_step': -1,
 'keyframes': False,
 'scene_bbox': [[-5.5, -5.5, -5.5], [5.5, 5.5, 5.5]],

 # Optimization settings
 'num_steps': 28001,
 'batch_size': 4096,
 'scheduler_type': 'warmup_cosine',
 'optim_type': 'adam',
 'lr': 0.003,

 # Regularization
 'distortion_loss_weight': 0.008,
 'histogram_loss_weight': 1.0,
 'l1_time_planes': 0.0005,
 'l1_time_planes_proposal_net': 0.0005,
 'plane_tv_weight': 0.0001,
 'plane_tv_weight_proposal_net': 0.0001,
 'time_smoothness_weight': 0.1,
 'time_smoothness_weight_proposal_net': 0.001,

 # Training settings
 'valid_every': 28000,
 'save_every': 28000,
 'save_outputs': True,
 'train_fp16': True,

 # Raymarching settings
 'single_jitter': False,
 'num_samples': 48,
 'num_proposal_iterations': 2,
 'num_proposal_samples': [256, 128],
 'use_same_proposal_network': False,
 'use_proposal_weight_anneal': True,
 'proposal_net_args_list': [
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [64, 64, 64, 100]},
  {'num_input_coords': 4, 'num_output_coords': 8, 'resolution': [128, 128, 128, 100]}
 ],

 # Model settings
 'concat_features_across_scales': True,
 'multiscale_res': [1, 2, 4, 8],
 'density_activation': 'trunc_exp',
 'linear_decoder': True,
 'linear_decoder_layers': 4,
 # Use time reso = half the number of frames
 # Lego: 25 (50 frames)
 # Hell Warrior and Hook: 50 (100 frames)
 # Mutant, Bouncing Balls, and Stand Up: 75 (150 frames)
 # T-Rex	and Jumping Jacks: 100 (200 frames)
 # zjuhuman: 115 (230 frames)
 'grid_config': [{
  'grid_dimensions': 2,
  'input_coordinate_dim': 4,
  'output_coordinate_dim': 32,
 # 'resolution': [64, 64, 64, 100] # original
  'resolution': [128, 128, 128, 100] # new test
 }],
}