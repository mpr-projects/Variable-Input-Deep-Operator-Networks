import h5py


class hdf5Writer:
    """Helper class to save output."""

    def __init__(self, fpath):
        fpath += '' if fpath[-5:] == '.hdf5' else '.hdf5'
        self.f = h5py.File(fpath, 'w')
        self.datasets = dict()

    def close(self):
        if self.f:
            self.f.close()

    def __del__(self):
        self.close()

    def register_group(self, name, parent=None):
        if parent is None:
            parent = self.f

        self.datasets[name] = parent.create_group(name)

    def add_data(self, name, value, parent=None):

        if parent is None:
            ds_name = name
            parent = self.f

        else:
            ds_name = parent + '/' + name
            parent = self.datasets[parent]

        self.datasets[ds_name] = parent.create_dataset(name, value.shape)
        self.datasets[ds_name][...] = value

    def add_meta(self, key, value, parent=None):
        parent = self.f if parent is None else self.datasets[parent]
        parent.attrs[key] = value

    def get_meta(self, key, parent=None):
        parent = self.f if parent is None else self.datasets[parent]
        return parent.attrs[key]
