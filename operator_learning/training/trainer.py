import torch
import time
from contextlib import suppress


class DefaultTrainer():
    """
    Train a neural network with the given parameters and settings.

    It assumes all objects are already on the correct device.

    """
    def __init__(self, model, optimizer, closure, dataloader,
                 output_dir='.', max_epochs=500, device='cpu'):
        """Set up trainer."""
        self.model = model
        self.optimizer = optimizer
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.device = device

        # give closure access to model, optimizer and data
        self.closure = closure

        self.start_of_epoch_hooks = []
        self.end_of_epoch_hooks = [self.closure]

        self.stop_training = False  # to be set by hooks

    def attach_hooks(self, *hooks, where='end'):
        """
        Attach one or many hook(s) to the trainer.

        Each hook must be a subclass of 'BaseHook'. In every
        epoch the class member 'step' is called. It is passed the
        parameter 'eid' which is the running index of the current
        epoch. The hook class has access to the state of the trainer
        through the member function 'self._trainer'.

        Parameter 'where' can be either 'start' or 'end'. It
        determines whether the hook is called at the beginning or
        at the end of the epoch.
        """

        assert where in ['start', 'end'], \
            f"Invalid value ('{where}') provided for argument 'where'."

        if where == 'start':
            self.start_of_epoch_hooks += list(hooks)

        else:
            self.end_of_epoch_hooks += list(hooks)
            
    @property
    def hooks(self):
        return self.start_of_epoch_hooks + self.end_of_epoch_hooks

    def set_trainer_in_hooks(self):
        """Set attribute 'trainer' of each hook to this class."""

        for hook in self.start_of_epoch_hooks + self.end_of_epoch_hooks:
            hook.trainer = self

    def notify_hooks_of(self, what):
        """
        Notify hooks of begining or end of training

        If parameter 'what' is 'start' then the hooks' member function
        'start_training' is called. If it is 'end' then the member
        function 'end_training' is called.
        """

        for hook in self.start_of_epoch_hooks + self.end_of_epoch_hooks:
            if what == 'start':
                with suppress(AttributeError):  # for lr_schedulers
                    hook.start_training()

            elif what == 'end':
                with suppress(AttributeError):  # for lr_schedulers
                    hook.stop_training()

    def run(self, start_time=None, print_output=True):
        """Run training process for up to 'max_epochs' epochs."""

        self.start_time = start_time or time.time()
        self.model.train()

        self.set_trainer_in_hooks()
        self.notify_hooks_of('start')

        # not using a for loop so max_epochs can be changed from
        # outside the training process
        self.eid = 0  

        while self.eid < self.max_epochs:
            self.loss = torch.tensor([0.], device=self.device)

            for hook in self.start_of_epoch_hooks:
                hook.step()

            n_batches = 0

            for data in self.dataloader:
                self._current_data = data
                self.loss += self.optimizer.step(self.closure).item()
                n_batches += 1

            self.loss /= n_batches

            for hook in self.end_of_epoch_hooks:
                hook.step()

            if self.stop_training:
                break

            self.eid += 1

        self.notify_hooks_of('end')
        print('\n')  # new line so no potential output gets overwritten
