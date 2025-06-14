import torch
from hooks import BaseHook


class ClosureBase(BaseHook):
    """
    Base class for closures.
    
    The main task of the base class is to provide setters and getters
    for the trainer attribute (via BaseHook). This variable is used to
    access the trainer and its state.
    
    """
    def __init__(self):
        super().__init__()


class DefaultSupervisedClosure(ClosureBase):
    """
    Default closure for optimizer for supervised training.

    The class is initialized by the user with their loss function of
    choice. The trainer later sets the property 'trainer' such that
    the closure obtains access to the trainer's state.

    The loss is calculated by applying the loss function to the
    model's prediction and the target value.

    """
    def __init__(self, loss_fn, reshape_fn=None, output_transform_fn=None):
        """Set loss function to be used in training."""
        super().__init__()
        
        self.loss_fn = loss_fn
        self.reshape_fn = reshape_fn
        self.output_transform_fn = output_transform_fn
        
        if reshape_fn is None:
            self.reshape_fn = lambda x: x
            
        if output_transform_fn is None:
            self.output_transform_fn = lambda x, y: y

    def __call__(self):
        """Apply pre-set loss function to model and data set in trainer."""
        inputs, targets = self.reshape_fn(self.trainer._current_data)

        if torch.is_grad_enabled():
            self.trainer.optimizer.zero_grad()

        pred = self.trainer.model(inputs)
        pred = self.output_transform_fn(inputs, pred)
        loss = self.loss_fn(pred, targets)

        if loss.requires_grad:
            loss.backward()

        return loss
