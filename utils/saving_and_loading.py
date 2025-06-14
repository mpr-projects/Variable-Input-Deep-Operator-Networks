import torch

# -------------------------------------------------------------------------------


def save_model(model, model_creation_fn, model_creation_params, save_path):
    """
    Save model and model creation parameters for later loading.
    
    To easily load trained models we use functions for the creation of models.
    Such a functions accepts the parameters required by each of the components of
    the final model and returns the new model and the parameters used in its
    creation. For example,
    
    Example:
    
        >>> import torch

        >>> def create_simple_model(nNeurons):
        >>>     return torch.nn.Linear(1, nNeurons), dict(nNeurons=nNeurons)

        >>> model, model_creation_params = create_simple_model(10)

        >>> # train and save model and model_creation_params, at a later time:

        >>> model, _ = create_simple_model(**model_creation_params)
        
    This functions saves the model's state dict along with the name of the model
    creation function and the required parameters.
    
    """
    save_dict = dict(
        model=model.state_dict(),
        model_creation_fn=model_creation_fn,
        model_creation_params=model_creation_params)
    
    torch.save(save_dict, save_path)
    
    
def load_model(load_path, module=None, creation_fn=None, device=None):
    """
    Load model previously saved with 'save_model'.
    
    The function that was used to create the model must either be passed as
    parameter 'creation_fn' or the module where it is defined must be passed
    as 'module'.
    
    """
    assert module is not None or creation_fn is not None, \
        "Either 'module' or 'creation_fn' must be provided."
        

    load_params = {} if device is None else {'map_location': device}
    load_dict = torch.load(load_path, **load_params)
    model_fn_name = load_dict['model_creation_fn']
    model_params = load_dict['model_creation_params']

    if device is not None:
        model_params['device'] = device
    
    if module is None:
        assert creation_fn.__name__ == model_fn_name, (
            "Name of creation_fn differs from function name used to create" 
            " the model.")
        
        # model, _ = creation_fn(**model_params)
        model = creation_fn(**model_params)

        if isinstance(model, list) or isinstance(model, tuple):
            model, _ = model
        
    else:
        model, _ = getattr(module, model_fn_name)(**model_params)
        
    model.load_state_dict(load_dict['model'])
    return model


def load_state_dict(load_path):
    """
    Load state dict of saved model.
    
    """
    load_dict = torch.load(load_path)
    return load_dict['model']
