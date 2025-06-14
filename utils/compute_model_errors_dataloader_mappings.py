from saving_and_loading import load_model

from deeponets import MLPDeepONet
from vidon import Vidon
from fourier_2d import FNO2d

import hdf5_dataloaders_deeponet
import hdf5_dataloaders_vidon
import hdf5_dataloaders_fno


def get_dataloader_MLPDeepONet(name, model_file, data_file, settings, inputs_hook):
    model = load_model(model_file, creation_fn=MLPDeepONet, device='cpu')

    return model, hdf5_dataloaders_deeponet.CreateDeepONetDataloaderWithoutPreloading(
        data_file,
        batch_size=settings.get(f'batch_size_test', 1),
        inputs_hook=inputs_hook)


def get_dataloader_Vidon(name, model_file, data_file, settings, inputs_hook):
    model = load_model(model_file, creation_fn=Vidon, device='cpu')

    dl_types = {
        'default': hdf5_dataloaders_vidon.CreateVidonDataloaderDefaultFormat,
        'vidon': hdf5_dataloaders_vidon.CreateVidonDataloader
    }

    dl_train_type = dl_types[settings.get(f'format_{name}_file', 'vidon')]

    return  model, dl_train_type(
        data_file,
        batch_size=settings.get(f'batch_size_test', None),
        inputs_hook=inputs_hook)


def get_dataloader_FNO2d(name, model_file, data_file, settings, inputs_hook):
    model = load_model(model_file, creation_fn=FNO2d, device='cpu')

    return model, hdf5_dataloaders_fno.CreateFNODataloader(
        data_file,
        batch_size=settings.get(f'batch_size_test', None),
        inputs_hook=inputs_hook)


dataloader_mapping = {
    "MLPDeepONet": get_dataloader_MLPDeepONet,
    "Vidon": get_dataloader_Vidon,
    "FNO2d": get_dataloader_FNO2d,
}


get_normalizer_mapping = {
    "MLPDeepONet": hdf5_dataloaders_deeponet.get_normalizer,
    "Vidon": hdf5_dataloaders_vidon.get_normalizer,
    "FNO2d": hdf5_dataloaders_fno.get_normalizer
}
