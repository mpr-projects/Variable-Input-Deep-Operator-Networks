import os
import time

from trainer import DefaultTrainer


class DefaultRetrainer():
    """
    Train the same model multiple times with different model parameter initializations.

    """
    def __init__(self, model_type, model_params, optimizer_type, optimizer_params, closure, dataloader,
                 output_dir='.', max_epochs=500, trainer=DefaultTrainer, skip_indices=[], device='cpu'):

        self.model_type = model_type
        self.model_params = model_params
        self.optimizer_type = optimizer_type
        self.optimizer_params = optimizer_params
        self.closure = closure
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.max_epochs = max_epochs
        self.trainer = trainer
        self.skip_indices = skip_indices
        self.device=device

        self.start_of_epoch_hooks = []
        self.end_of_epoch_hooks = []

    def attach_hooks(self, *hooks, where='end'):
        """
        Attach one or many hook(s) to the trainer.

        """
        assert where in ['start', 'end'], \
            f"Invalid value ('{where}') provided for argument 'where'."

        if where == 'start':
            self.start_of_epoch_hooks += list(hooks)

        else:
            self.end_of_epoch_hooks += list(hooks)

    def run(self, n_retrainings=1, print_output=True):

        for rid in range(n_retrainings):
            model = self.model_type(**self.model_params)
            model.to(self.device)

            if rid in self.skip_indices:
                continue

            if print_output:
                model.print_size()

            optimizer = self.optimizer_type(
                model.parameters(), **self.optimizer_params)

            # get name of output directory
            start_time = time.time()
            start_time_fmt = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime(start_time))
            output_dir = os.path.join(self.output_dir, f'run_{rid}_{start_time_fmt}')
            os.makedirs(output_dir)

            # create link to settings file (used by error calculation)
            json_files = [f for f in os.listdir(self.output_dir) if f[-5:] == '.json']

            for json_file in json_files:
                target_file = os.path.join(output_dir, json_file)
                os.system(f'ln -s ../{json_file} {target_file}')

            # run training
            trainer = self.trainer(
                model, optimizer, self.closure, self.dataloader,
                output_dir=output_dir, max_epochs=self.max_epochs, device=self.device)

            trainer.attach_hooks(*self.start_of_epoch_hooks, where='start')
            trainer.attach_hooks(*self.end_of_epoch_hooks, where='end')

            trainer.run(start_time=start_time, print_output=print_output)
