import sys
sys.path.insert(0, "./original/")

import pytorch_lightning as pl
import models
import data_modules

import inversefed

def attack(model, loss_fn, data_module, rlabel=True, rec_machine_optimizer=None):
	''' Reconstruction algorithm'''
	# get data (ground_truth, labels, data_mean, data_std)
	data_mean = data_module.mean
	data_std = data_module.std
	num_images = data_module.batch_size
	shape = data_module.dims

	data_module.prepare_data()
	data_loader = data_module.val_dataloader()
	ground_truth, labels = next(iter(data_loader))

	# get target_loss
	model.zero_grad()
	target_loss, _, _ = loss_fn(model(ground_truth), labels)

	# get input_gradient
	input_gradient = torch.autograd.grad(target_loss, model.parameters())

	# reconstruct
	config = create_config(rec_machine_optimizer)
	rec_machine = inversefed.GradientReconstructor(model, (data_mean, data_std), config, num_images)
	if rlabel:
	    output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape) # reconstruction label
	else:
	    output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape) # specify label

	# denormalize
	output_denormalized = output * data_std + data_mean
	input_denormalized = ground_truth * data_std + data_mean

	# calculate metrics
	test_mse = (output_denormalized - input_denormalized).pow(2).mean()
	feat_mse = (model(output) - model(ground_truth)).pow(2).mean()
	test_psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

	metrics = (test_mse, feat_mse, test_psnr)

	return output_denormalized, metrics

# original code to get a configuration dict for RecMachine (maybe turn this into something prettier)
def create_config(optimizer):
    if optimizer == 'inversed':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-4,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')
    elif optimizer == 'inversed-zero':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-4,
                      init='zeros',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')
    elif optimizer == 'inversed-sim-out':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='out_sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-4,
                      init='zeros',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')
    elif optimizer == 'inversed-sgd-sim':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='sgd',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-4,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')
    elif optimizer == 'inversed-LBFGS-sim':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=1e-4,
                      optim='LBFGS',
                      restarts=16,
                      max_iterations=300,
                      total_variation=1e-4,
                      init='randn',
                      filter='none',
                      lr_decay=False,
                      scoring_choice='loss')
    elif optimizer == 'inversed-adam-L1':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='l1',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-4,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')
    elif optimizer == 'inversed-adam-L2':
        config = dict(signed=True,
                      boxed=True,
                      cost_fn='l2',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-4,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')
    elif optimizer == 'zhu':
        config = dict(signed=False,
                      boxed=False,
                      cost_fn='l2',
                      indices='def',
                      weights='equal',
                      lr=1e-4,
                      optim='LBFGS',
                      restarts=2,
                      max_iterations=50,  # ??
                      total_variation=1e-3,
                      init='randn',
                      filter='none',
                      lr_decay=False,
                      scoring_choice='loss')
    else: # InverseFed DEFAULT
    	config = dict(signed=False,
                      boxed=True,
                      cost_fn='sim',
                      indices='def',
                      weights='equal',
                      lr=0.1,
                      optim='adam',
                      restarts=1,
                      max_iterations=4800,
                      total_variation=1e-1,
                      init='randn',
                      filter='none',
                      lr_decay=True,
                      scoring_choice='loss')
    return config