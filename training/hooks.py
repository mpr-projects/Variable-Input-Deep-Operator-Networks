import os
import ast
import time
import torch
import numpy as np
from collections import deque

from saving_and_loading import save_model, load_state_dict
from error_calculation import compute_error, relative_Lp_error


class BaseHook():
    """
    Base class for all hooks.

    To create a custom hook create a subclass of 'BaseHook'.
    You can overwrite methods 'start_training', 'step' and
    'stop_training' as appropriate for your custom hook.

    """

    def __init__(self):
        # we use a mutable object to simulate a c-type pointer;
        # this is done so that the validation loss hook can set
        # the trainer of its closure to its own trainer before
        # the trainer has set the trainer variable; otherwise the
        # user would have to explicitly set the trainer when
        # creating a different validation closure to the training
        # closure; since this variable is generally not visible
        # to the user they should not be required to do so in
        # this case either
        self._trainer = [None]

    @property
    def trainer(self):
        return self._trainer[0]

    @trainer.setter
    def trainer(self, trainer):
        self._trainer[0] = trainer

    def remove(self):
        if self in self.trainer.start_of_epoch_hooks:
            self.trainer.start_of_epoch_hooks.remove(self)

        if self in self.trainer.end_of_epoch_hooks:
            self.trainer.end_of_epoch_hooks.remove(self)

    def start_training(self):
        """Trainer calls this function at the beginning of training."""
        pass

    def step(self):
        """Trainer calls this function once every epoch."""
        pass

    def stop_training(self):
        """Trainer calls this function at the end of training."""
        pass


# -------------------------------------------------------------------------------
# Interactive Hooks
# -------------------------------------------------------------------------------
def attach_interactive_hooks(trainer):
    trainer.attach_hooks(
        ListHooksHook(),
        ManualStoppingHook(),
        ManuallyChangeLearningRateHook(),
        ManuallyChangeTrainerPropertyHook(),
        ManuallyRemoveHook()
    )


class InteractiveBaseHook(BaseHook):
    """
    Base class for all hooks which allow the user to affect the training process
    when it's already running.

    """
    def __init__(self, frequency, delete_file, file_name):
        super().__init__()
        self.frequency = frequency
        self.delete_file = delete_file
        self.file_name = file_name
        self.pending_changes = dict()

    def _delete_file(self, file_name):
        if self.delete_file:
            os.remove(file_name)

    def step(self):
        if self.trainer.eid in self.pending_changes:
            self._step(pending_change=True)
            return

        if self.trainer.eid % self.frequency != 0:
            return

        if os.path.exists(self.file_name):
            self._step()
            self._delete_file(self.file_name)
            return

        start_time = time.strftime(
            '%Y-%m-%d_%H-%M-%S',
            time.localtime(self.trainer.start_time)
        )

        file_name = f'{self.file_name}_{start_time}'

        if os.path.exists(file_name):
            self._step(file_name=file_name)
            self._delete_file(file_name)


class ManuallyChangeLearningRateHook(InteractiveBaseHook):
    """
    Set the learning rate from outside the training process.

    This hook looks for a file called SET_LR. The first entry in this
    file will be taken as new learning rate for all parameter groups. If there is
    only one entry then the change will be made immediately. Otherwise the second
    entry must be the epoch when the change should be made.

    """
    def __init__(self, frequency=10, delete_file=True, file_name='SET_LR'):
        super().__init__(frequency, delete_file, file_name)

    def _set_lr(self, lr):
        for pg in self.trainer.optimizer.param_groups:
            pg['lr'] = lr

        print("\n\nManually changed learning rate.\n\n")

    def _step(self, pending_change=False, file_name=None):
        if pending_change:
            new_lr = self.pending_changes.pop(self.trainer.eid)
            self._set_lr(new_lr)
            return

        with open(file_name or self.file_name) as f:
            contents = f.readline().rstrip().split()

        try:
            new_lr = float(contents[0])

        except ValueError:
            print(f'\n\nInvalid learning rate ({contents[0]}) set in file {self.file_name}.\n\n')
            return

        if len(contents) == 1:
            self._set_lr(new_lr)
            return

        if len(contents) != 2:
            print(f'\n\nInvalid number of entries ({len(contents)} in file {self.file_name}.')

        try:
            epoch = int(contents[1])

        except ValueError:
            print(f'\n\nInvalid epoch ({contents[1]}) set in file {self.file_name}.\n\n')
            return

        self.pending_changes[epoch] = new_lr


class ManuallyChangeTrainerPropertyHook(InteractiveBaseHook):
    """
    Set the value of any attribute of the trainer from outside the training process.

    The first entry of the file called 'file_name' must be the attribute, the
    second must be its new value.

    """
    def __init__(self, frequency=10, delete_file=True, file_name='CHANGE_PROPERTY'):
        super().__init__(frequency, delete_file, file_name)

    def _step(self, file_name=None):
        with open(file_name or self.file_name) as f:
            line = f.readline().rstrip().split()

        attr = line[0]
        value = ast.literal_eval(line[1])
        setattr(self.trainer, attr, value)

        print(f"\n\nManually set attribute {line[0]} to {line[1]}.\n\n")


class ListHooksHook(InteractiveBaseHook):
    """
    List available hooks with their indices.

    """
    def __init__(self, frequency=10, delete_file=True, file_name='LIST_HOOKS'):
        super().__init__(frequency, delete_file, file_name)

    def _step(self, **kwargs):
        print('\n\nPre-Hooks:')

        for hid, hook in enumerate(self.trainer.start_of_epoch_hooks):
            print(f'{hid}: {hook}')

        print('\nPost-Hooks:')

        for hid, hook in enumerate(self.trainer.end_of_epoch_hooks):
            print(f'{hid}, {hook}')

        print('')



class ManuallyRemoveHook(InteractiveBaseHook):
    """
    Remove Hook with id found through ListHooksHook.

    File must contain either "pre/post hook_id" or "pre/post hook_id epoch" in first line.

    """
    def __init__(self, frequency=10, delete_file=True, file_name='REMOVE_HOOK'):
        super().__init__(frequency, delete_file, file_name)

    def _apply(self, pre_post, hook_id):
        if pre_post == 'pre':
            del self.trainer.start_of_epoch_hooks[hook_id]

        else:
            del self.trainer.end_of_epoch_hooks[hook_id]

    def _step(self, pending_change=False, file_name=None):
        if pending_change:
            pre_post, hook_id = self.pending_changes.pop(self.trainer.eid)
            self._apply(pre_post, hook_id)
            return

        with open(file_name or self.file_name) as f:
            contents = f.readline().rstrip().split()

        if len(contents) not in [2, 3]:
            print(f"\n\nTwo or three arguments must be provided in file {self.file_name}.\n\n")
            return

        pre_post = contents[0]

        if pre_post not in ['pre', 'post']:
            print(f"\n\nFirst value in file {self.file_name} must be in ['pre', 'post'].\n\n")
            return

        try:
            hook_id = int(contents[1])

        except ValueError:
            print(f"\n\nInvalid hook_id ({contents[1]}) set in file {self.file_name}.\n\n")
            return

        if len(contents) == 2:
            self._apply(pre_post, hook_id)
            return

        try:
            epoch = int(contents[2])

        except ValueError:
            print(f"\n\nInvalid epoch ({contents[2]}) set in file {self.file_name}.\n\n")
            return

        self.pending_changes[epoch] = (pre_post, hook_id)


# -------------------------------------------------------------------------------
# Unassigned Hooks
# -------------------------------------------------------------------------------
class RemoveHook(BaseHook):
    """
    Remove a given hook at a given epoch.

    """
    def __init__(self, hook, epoch):
        super().__init__()
        self.hook = hook
        self.epoch = epoch

    def step(self):
        if self.trainer.eid != self.epoch:
            return

        if self.hook in self.trainer.start_of_epoch_hooks:
            self.trainer.start_of_epoch_hooks.remove(self.hook)

        if self.hook in self.trainer.end_of_epoch_hooks:
            self.trainer.end_of_epoch_hooks.remove(self.hook)

        if self in self.trainer.start_of_epoch_hooks:
            self.trainer.start_of_epoch_hooks.remove(self)

        if self in self.trainer.end_of_epoch_hooks:
            self.trainer.end_of_epoch_hooks.remove(self)





# -------------------------------------------------------------------------------
# Error Calculation Hooks
# -------------------------------------------------------------------------------
class CalculateValidationLossHook(BaseHook):
    """
    Calculate loss on validation data.

    This hook sets an attribute 'validation_loss' in the Trainer to
    which it is linked.
    
    Parameter 'dataloader' must be passed the dataloader containing the
    validation data. Parameter 'closure' refers to the closure called to
    obtain the loss. It can be different than the closure used for training.
    This is required, for example, in semi-supervised training, where the
    training data consists of both supervised and unsupervised data but
    validation is typically only done on supervised data.
    
    If the validation loss shall not be calculated in every epoch then
    parameter 'frequency' must be set.

    """

    def __init__(self, dataloader, closure, reshape_fn=None, frequency=1):
        super().__init__()
        self.dataloader = dataloader
        self.frequency = frequency
        
        self.closure = closure
        # we use a mutable object to simulate a c-type pointer;
        # this is done so that the validation loss hook can set
        # the trainer of its closure to its own trainer before
        # the trainer has set the trainer variable; otherwise the
        # user would have to explicitly set the trainer when
        # creating a different validation closure to the training
        # closure; since this variable is generally not visible
        # to the user they should not be required to do so in
        # this case either
        self.closure._trainer = self._trainer
        self.reshape_fn = reshape_fn
        
        if reshape_fn is None:
            self.reshape_fn = lambda x: x
            
    def start_training(self):
        self.closure.start_training()

    def step(self):
        if self.trainer.eid % self.frequency != 0:
            return

        self.trainer.model.eval()
        self.trainer.validation_loss = torch.tensor(
                [0.], device=self.trainer.device)

        for data in self.dataloader:
            self.trainer._current_data = self.reshape_fn(data)
            self.trainer.validation_loss += self.closure()

        self.trainer.validation_loss /= len(self.dataloader)
        self.trainer.model.train()
        
        self.closure.step()
            
    def stop_training(self):
        self.closure.stop_training()


class CalculateRelativeL2ErrorHook(BaseHook):
    """
    Calculate relative L2 error on a dataset.

    If you use this hook more then once then different names have to be set.

    """
    def __init__(self, dataloader, name='relative_L2_error',
                 reshape_fn=lambda x: x, output_transform=lambda x, y: y,
                 frequency=1, sample_dimensions=0, batch_dimension=0):

        super().__init__()
        self.dataloader = dataloader
        self.frequency = frequency
        self.name = name

        self.reshape_fn = reshape_fn
        self.output_transform = output_transform

        self.sample_dimensions = sample_dimensions
        self.batch_dimension = batch_dimension

    def step(self):
        if self.trainer.eid % self.frequency != 0:
            return

        self.trainer.model.eval()

        # not sure if this works for physics-informed training
        # with torch.no_grad():
        rel_l2_errors = compute_error(
            self.trainer.model,
            self.dataloader,
            relative_Lp_error(sample_dimensions=self.sample_dimensions,
                              batch_dimension=self.batch_dimension),
            reshape_fn=self.reshape_fn,
            output_transform=self.output_transform)

        setattr(self.trainer, self.name, rel_l2_errors.mean())
        self.trainer.model.train()


# -------------------------------------------------------------------------------
# Checkpoint Hook
# -------------------------------------------------------------------------------
class CheckpointHook(BaseHook):
    """
    Save model on improvement.

    Parameter 'attribute' is the name of the attribute in the Trainer which
    is used to evaluate improvement. Parameter 'decreasing' says if the
    attribute should decrease or increase. The path where checkpoints are
    saved can be set with 'output_dir'. The file format of the saved
    chekcpoints is given by 'file_format'. It can contain the variables
    'start_time', 'eid' and 'attr'. 'attr' refers to the attribute used for
    evaluation. Parameter 'n_saved' determines how many checkpoints will be
    saved.

    """

    def __init__(self, attribute, model_creation_fn, model_creation_params,
                 decreasing=True, n_saved=1,
                 prefix='', file_format='{start_time}_{eid:0>{len_e}}_{attr:.4e}.pt',
                 load_best=False, delete_models=False):
        """
        If 'load_best' is set to true then the best model found during training
        will be loaded at the end of the training process.
        
        """
        super().__init__()
        self.attribute = attribute
        self.model_creation_fn = model_creation_fn
        self.model_creation_params = model_creation_params
        self.decreasing = decreasing
        self.file_format = file_format
        self.prefix = prefix
        self.load_best = load_best
        self.delete_models = delete_models

        self.n_saved = n_saved
        self.saved_models = deque()

        if decreasing:
            self.step = self._step_decreasing
        else:
            self.step = self._step_increasing

    def start_training(self):
        self.best_value = float('inf')
        self.saved_models = deque()

        if not self.decreasing:
            self.best_value *= -1

        self.output_dir = os.path.join(self.trainer.output_dir, 'checkpoints')
        os.makedirs(self.output_dir, exist_ok=True)

        # prefix  may be changed after initialization (eg by retrainer),
        # --> update every time we save the model
        self.prefix_file_format = self.prefix + self.file_format

    def _step_decreasing(self):
        if getattr(self.trainer, self.attribute) < self.best_value:
            self.best_value = getattr(self.trainer, self.attribute).item()
            self._save_model()

    def _step_increasing(self):
        if getattr(self.trainer, self.attribute) > self.best_value:
            self.best_value = getattr(self.trainer, self.attribute).item()
            self._save_model()

    def _save_model(self):
        start_time = time.strftime(
                '%Y-%m-%d_%H-%M-%S',
                time.localtime(self.trainer.start_time)
        )

        len_e = len(str(self.trainer.max_epochs))

        path = os.path.join(self.output_dir, self.prefix_file_format.format(
            start_time=start_time, eid=self.trainer.eid, len_e=len_e, attr=self.best_value))

        self.saved_models.appendleft(path)

        if len(self.saved_models) > self.n_saved:
            os.remove(self.saved_models.pop())

        save_model(
            self.trainer.model, self.model_creation_fn,
            self.model_creation_params, path)
        
    def stop_training(self):
        # always save last model of training process
        start_time = time.strftime(
                '%Y-%m-%d_%H-%M-%S',
                time.localtime(self.trainer.start_time)
        )

        len_e = len(str(self.trainer.max_epochs))

        path = os.path.join(self.output_dir, self.prefix_file_format.format(
            start_time=start_time, eid=self.trainer.eid, len_e=len_e, attr=self.trainer.loss.item()))

        path = path[:-3] + '_loss.pt'

        save_model(
            self.trainer.model, self.model_creation_fn,
            self.model_creation_params, path)
        
        if self.load_best:
            self.trainer.model.load_state_dict(
                load_state_dict(self.saved_models[0]))

        if self.delete_models:
            for model in self.saved_models:
                os.remove(model)

            self.saved_model = deque()


# -------------------------------------------------------------------------------
# Early Stopping Hooks
# -------------------------------------------------------------------------------
class ManualStoppingHook(InteractiveBaseHook):
    """
    Stop training if a certain file is present.

    This hook allows to stop training in an ordered fashion. The user needs to
    place a file called 'STOP_TRAINING_PROCESS' in the directory where training
    was started. By default this file will be deleted when training stops.

    It's best to put the hook last in the list of hooks so that the output on
    stopping training is not overwritten by other hooks.

    """
    def __init__(self, frequency=10, delete_file=True,
                 file_name='STOP_TRAINING_PROCESS'):
        super().__init__(frequency, delete_file, file_name)

    def _step(self, pending_change=False, file_name=None):
        if pending_change:
            self.trainer.stop_training = True
            return

        with open(file_name or self.file_name) as f:
            contents = f.readline().rstrip().split()

        if len(contents) == 0:
            self.trainer.stop_training = True
            return

        if len(contents) != 1:
            print(f'\n\nInvalid number of entries ({len(contents)} in file {self.file_name}.')

        try:
            epoch = int(contents[0])

        except ValueError:
            print(f'\n\nInvalid epoch ({contents[0]}) set in file {self.file_name}.\n\n')
            return

        self.pending_changes[epoch] = 0  # dummy value


class EarlyStoppingHook(BaseHook):
    """
    Stop training if a given attribute stops improving.

    Parameter 'attribute' sets the member of the Trainer to be used and
    parameter 'decreasing' determines if an improvement is an increase or
    a decrease in the value of the attribute.
    Parameter 'min_delta' refers to the minimum improvement necessary for
    it to count as improvement. Parameter 'percentage' sets if 'min_delta'
    is an absolute value or a percentage of the previously best value.
    Parameter 'patience' determines after how many epochs without
    improvement training is stopped.

    """

    def __init__(self, attribute, patience, decreasing=True,
                 min_delta=0, percentage=True):
        super().__init__()
        self.attribute = attribute
        self.patience = patience
        self.min_delta = min_delta
        self.decreasing = decreasing

        if decreasing is True and percentage is False:
            def improvement():
                return getattr(self.trainer, self.attribute) \
                    < self.best_value - self.min_delta

        elif decreasing is True and percentage is True:
            def improvement():
                return getattr(self.trainer, self.attribute) \
                    < self.best_value * (1 - self.min_delta)

        elif decreasing is False and percentage is False:
            def improvement():
                return getattr(self.trainer, self.attribute) \
                    > self.best_value + self.min_delta

        else:
            def improvement():
                return getattr(self.trainer, self.attribute) \
                    > self.best_value * (1 + self.min_delta)

        self._improvement = improvement

    def start_training(self):
        self.count = 0
        self.best_value = float('inf')

        if not self.decreasing:
            self.best_value *= -1

    def step(self):
        if self._improvement():
            self.best_value = getattr(self.trainer, self.attribute)
            self.count = 0
        else:
            self.count += 1

        if self.count >= self.patience:
            print('\nReached early stopping limit.')
            self.trainer.stop_training = True


class TimeLimitHook(BaseHook):
    """
    Stop training after fixed time (in seconds).

    """

    def __init__(self, limit=300, verbose=True):
        super().__init__()
        self.limit = limit
        self.verbose = verbose

    def start_training(self):
        self.start_time = time.time()

    def step(self):
        if time.time() - self.start_time > self.limit:
            if self.verbose:
                print('\nReached time limit.')

            self.trainer.stop_training = True


class StopOnNaNHook(BaseHook):
    """
    Stop training if the loss becomes NaN.

    Example:

    >>> print('Hello')
    StopOnNaNHook

    """

    def __init__(self, verbose=True):
        super().__init__()
        self.verbose = verbose

    def step(self):
        if torch.isnan(self.trainer.loss).any():
            if self.verbose:
                print("\nStopping because of NaN.")

            self.trainer.stop_training = True


# -------------------------------------------------------------------------------
# Batch Size Scheduler Hooks
# -------------------------------------------------------------------------------
class LRSchedulerHook(BaseHook):
    """
    Change learning rate with built-in PyTorch LR-schedulers.

    """
    def __init__(self, scheduler_type, params):
        super().__init__()
        self.scheduler_type = scheduler_type
        self.params = params

    def start_training(self):
        self.scheduler = self.scheduler_type(
            self.trainer.optimizer, **self.params)

    def step(self):
        self.scheduler.step()


# -------------------------------------------------------------------------------
# Output Hooks
# -------------------------------------------------------------------------------
class PrintHook(BaseHook):
    """
    Print information on training progress every epoch.

    """

    def __init__(self, decimals=3, scalars={'loss': 'loss'},
                 lr=True, bs=True, new_line=False):
        """The values passed to 'scalars' must be members of the trainer."""
        
        super().__init__()
        self.decimals = decimals
        end = '\n' if new_line else '              \r'
        
        string = "print(f\"epoch {self.trainer.eid} "
        
        if lr or bs:
            string += "("
            
        if lr:
            string += "lr={self.trainer.optimizer.param_groups[0]['lr']:.3e}"
            
        if lr and bs:
            string += ", "
            
        if bs:
            string += "bs={self.trainer.dataloader.batch_size}"
            
        if lr or bs:
            string += ")"
            
        string += ": "
        
        for k, v in scalars.items():
            string += (
                f"{k}={{getattr(self.trainer,"
                f" '{v}').item():.{decimals}e}} ")
            
        string = string[:-1] + "\""
        string += ", end=\""
        string += "\\n" if new_line else "               \\r"
        string += "\")"
        self.code = compile(string, '<string>', 'eval')
        
    def step(self):
        eval(self.code)


class TensorBoardHook(BaseHook):
    """
    Write data to TensorBoard.

    By default only the training loss will be written to TensorBoard. However,
    you can save any member of the Trainer class by setting parameter
    'attributes' during initialization. In particular, 'attributes' has to be
    a dict whose keys are used as TensorBoard 'tags' and whose values will be
    saved with TensorBoard's 'add_scalar'.

    """

    def __init__(self, name='run', add_time=False,
                 attributes={'training_loss': 'loss'}):
        """
        Initialize TensorBoard hook.

        If parameter 'add_time' is set then the start time of the training
        process will be appended to the 'runs_directory'.
        """

        super().__init__()
        self._writer = None
        self.name = name
        self.add_time = add_time
        self.attributes = attributes

    def start_training(self):
        self.output_dir = os.path.join(self.trainer.output_dir, self.name)
        output_dir = self.output_dir

        if self.add_time:
            output_dir += time.strftime(
                '_%Y-%m-%d_%H-%M-%S',
                time.localtime(self.trainer.start_time)
            )

        from torch.utils.tensorboard import SummaryWriter
        self.writer = SummaryWriter(output_dir)

    def step(self):
        for name in self.attributes:
            self.writer.add_scalar(
                name,
                getattr(self.trainer, self.attributes[name]),
                global_step=self.trainer.eid
            )

    def stop_training(self):
        self.writer.close()
        self.writer = None


class SaveRuntimeHook(BaseHook):
    """
    Write runtime to file in output folder.

    """
    def __init__(self):
        super().__init__()

    def stop_training(self):
        runtime = time.time() - self.trainer.start_time
        hours, rest = divmod(runtime, 3600)
        minutes, seconds = divmod(rest, 60)

        file_name = os.path.join(self.trainer.output_dir, 'runtime.txt')

        with open(file_name, "w") as f:
            f.write(f"{runtime} seconds ({int(hours)}h {int(minutes)}min {int(round(seconds))}sec)")
