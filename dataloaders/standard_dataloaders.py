import torch


def DataLoader(dataset, batch_size=None, **kwargs):
    """
    DataLoader with full batch as default batch size.
    
    """        
    batch_size = len(dataset) if batch_size is None else batch_size
    kwargs.update({'batch_size': int(batch_size)})

    return torch.utils.data.DataLoader(dataset, **kwargs)
