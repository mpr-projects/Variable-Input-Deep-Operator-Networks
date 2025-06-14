import h5py
import json
import torch
import torch.nn as nn
import torch.optim.lr_scheduler
from torch.utils.data import TensorDataset, ConcatDataset

import numpy as np

import sys
sys.path.append('../../operator_learning/utils')
sys.path.append('../../operator_learning/training')
sys.path.append('../../operator_learning/models')
sys.path.append('../../operator_learning/dataloaders')
sys.path.append('../../operator_learning/datasets')

from retrainer import DefaultRetrainer
from closures import DefaultSupervisedClosure

from hooks import (
    attach_interactive_hooks,
    PrintHook,
    StopOnNaNHook,
    TimeLimitHook,
    TensorBoardHook,
    CheckpointHook,
    EarlyStoppingHook,
    CalculateRelativeL2ErrorHook,
    LRSchedulerHook,
    SaveRuntimeHook)

import normalizers
from fourier_2d import FNO2d
from standard_dataloaders import DataLoader
from hdf5_dataloaders_fno import CreateFNODataloader, get_normalizer
from parse_common_arguments import get_args


# settings
# -----------------------------------------------------------------------------
args = get_args()
device = args.device
output_dir = args.output_dir
print_output = not args.no_output

assert args.settings is not None, "You must provide a settings file."

with open(args.settings) as f:
    settings = json.load(f)


# set seed for reproducibility
# -----------------------------------------------------------------------------
torch.manual_seed(settings['rng_seed'])


# load training data
# -----------------------------------------------------------------------------
normalizer_inputs, _ = get_normalizer(
    settings['training_file'], settings.get('normalizer_inputs', None),
    'input', device)

normalizer_outputs, output_transform = get_normalizer(
    settings['training_file'], settings.get('normalizer_outputs', None),
    'output', device)

dataloader, reshape_fn_train = CreateFNODataloader(
    settings['training_file'],
    batch_size=settings['batch_size_training'],
    shuffle=True,
    inputs_hook=normalizer_inputs,
    outputs_hook=normalizer_outputs,
    device=device)


# load validation data for early stopping
# -----------------------------------------------------------------------------
dataloader_val, reshape_fn_val = CreateFNODataloader(
    settings['validation_file'],
    inputs_hook=normalizer_inputs,
    device=device)


# create model
# -----------------------------------------------------------------------------
model_params = settings['model_params']
model_params['device'] = device

model_type = FNO2d


# create closure
# -----------------------------------------------------------------------------
loss_fn = getattr(nn, settings.get('loss_fn', 'MSELoss'))()
closure = DefaultSupervisedClosure(loss_fn, reshape_fn=reshape_fn_train)


# create optimizer
# -----------------------------------------------------------------------------
optimizer_type = torch.optim.Adam


# learning rate scheduler
# -----------------------------------------------------------------------------
lr_scheduler_name = settings.get('lr_scheduler', None)

if lr_scheduler_name is not None:
    exec(f'lr_scheduler = torch.optim.lr_scheduler.{lr_scheduler_name}')

    
# train model
# -----------------------------------------------------------------------------
trainer = DefaultRetrainer(
    model_type, model_params, optimizer_type, settings['optimizer_params'],
    closure, dataloader, output_dir=output_dir,
    max_epochs=settings['max_epochs'], device=device)

checkpoint = CheckpointHook(
    'relative_L2_error', model_type.__name__, model_params, n_saved=1,
    prefix='fno_')

trainer.attach_hooks(
    CalculateRelativeL2ErrorHook(
        dataloader_val, reshape_fn=reshape_fn_val, output_transform=output_transform),
    TensorBoardHook(name='fno_rL2', 
                    attributes={'FNO': 'relative_L2_error'}),
    checkpoint,
    where='start')

if print_output:
    trainer.attach_hooks(
        PrintHook(scalars={'loss': 'loss', 'rL2': 'relative_L2_error'}))

if 'time_limit' in settings:
    trainer.attach_hooks(TimeLimitHook(limit=settings['time_limit']))

trainer.attach_hooks(
    StopOnNaNHook(),
    TensorBoardHook(name='fno_train', attributes={'FNO Loss': 'loss'}),
    # EarlyStoppingHook(attribute='loss', patience=1000),
    SaveRuntimeHook(),
)

if lr_scheduler_name is not None:
    trainer.attach_hooks(
        LRSchedulerHook(lr_scheduler, settings.get('lr_scheduler_params', {})))

attach_interactive_hooks(trainer)

trainer.run(settings.get('n_runs', 1))
