import torch


def compute_error(model, dataloader, error_fn,
                  dataloader_targets=None,
                  reshape_fn=lambda x: x,
                  output_transform=lambda x, y: y):
    """
    Compute the error of the model's predictions.
    
    If 'dataloader_targets' is None then the data from dataloader is assumed
    to be in format
    
        (inputs, targets)
        
    and inputs is passed to the 'reshape_fn' if provided. Otherwise, the
    'dataloader' is assumed to contain the inputs, which are passed to the
    'reshape_fn'. 'dataloader_targets' is assumed to contain the targets. Both
    dataloaders are expected to have samples in the same order (no shuffling).
    
    The model's predictions are passed to the 'output_transform'. Finally, the
    predictions and targets are passed to the 'error_fn', which is expected to
    return the appropriate error value.
    
    """
    if dataloader_targets is not None:
        dataloader = zip(dataloader, dataloader_targets)
        
    error_fn.reset()
        
    for data in dataloader:
        inputs, targets = reshape_fn(data)
        predictions = model(inputs)
        predictions = output_transform(inputs, predictions)

        error_fn.add(predictions, targets)

        # necessary if memory is too small to hold predictions
        # and targets at least twice
        del predictions
        del targets
        
    return error_fn.compute()


class relative_Lp_error_sum:
    """
    Not used for these experiments.

    Compute the relative Lp error when the batch dimension is different
    to the sample dimension.

    """
    def __init__(self, sample_dimensions, p):
        self.p = p
        self.sample_dim = sample_dimensions
        self.reset()

    def reset(self):
        self.diffs = 0
        self.targets = 0

    def add(self, predictions, targets):
        dim = tuple(i for i in range(targets.ndim) if i not in self.sample_dim)
        self.diffs += ((targets-predictions).abs()**self.p).sum(dim=dim).detach()
        self.targets += (targets.abs()**self.p).sum(dim=dim).detach()

    def compute(self):
        return (self.diffs / self.targets)**(1. / self.p)


class relative_Lp_error_app:
    """
    Compute the relative Lp error when the batch dimension is one of the
    sample dimensions.

    """
    def __init__(self, sample_dimensions, p):
        self.p = p
        self.sample_dim = sample_dimensions
        self.reset()

    def reset(self):
        self.res = []

    def add(self, predictions, targets):
        dim = tuple(i for i in range(targets.ndim) if i not in self.sample_dim)
        diffs = ((targets-predictions).abs()**self.p).sum(dim=dim).detach()
        targets = (targets.abs()**self.p).sum(dim=dim).detach()
        self.res.append((diffs / targets)**(1. / self.p))

    def compute(self):
        return torch.cat(self.res)


class relative_Lp_error:
    """
    Compute the relative Lp error.

    Errors are calculated over all dimensions except for the 'sample' dimension.

    """
    def __new__(cls, sample_dimensions, batch_dimension, p=2):
        """
        Return the appropriate class, depending on dimensions.

        """
        if not isinstance(batch_dimension, int):
            raise TypeError("'batch_dimension' must be an integer")

        if not isinstance(sample_dimensions, tuple):  # assume scalar
            sample_dimensions = (sample_dimensions,)

        if batch_dimension in sample_dimensions:
            return relative_Lp_error_app(sample_dimensions, p)

        return relative_Lp_error_sum(sample_dimensions, p)
