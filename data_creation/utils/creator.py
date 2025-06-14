from file_writers import hdf5Writer


class SampleCreator():
    """
    Create and save inputs and solutions to differential equations.

    """
    def __init__(self, settings, setup, get_input,  process_input,
                 solve, process_output, finish, writer=hdf5Writer):
        assert 'output_file' in settings, "Settings need to have an entry 'output_file'."
        self.writer = writer(settings['output_file'])
        self.writer.register_group('inputs')
        self.writer.register_group('outputs')

        for k in settings:
            self.writer.add_meta(k, settings[k])

        self.settings = settings
        self.state = dict()

        self.setup = setup
        self.get_input = get_input
        self.process_input = process_input
        self.solve = solve
        self.process_output = process_output
        self.finish = finish

    def run(self):
        self.setup(self)

        for sid in range(self.settings['n_samples']):
            self.sid = sid

            # the solve method can set 'valid_input' to False to draw
            # a new input (for whichever reason)
            while True:
                self.valid_input = True

                self.input = self.get_input(self)
                processed_input = self.process_input(self)

                self.output= self.solve(self)

                if self.valid_input:
                    break

            self.save_data(processed_input, 'inputs')

            processed_output = self.process_output(self)
            self.save_data(processed_output, 'outputs')

            print(f'finished {sid}', end=10*' '+'\r')

        self.finish(self)

        self.n_samples = None
        self.inputs = None
        self.solution = None

    def save_data(self, data, which):
        """
        Save data. 'data' must be a dict whose keys are the names
        of the datsets to be saved and whose values is the corresponding
        data. 'which' must be 'inputs' or 'outputs'.

        """
        for name in data:
            self.writer.add_data(name, data[name], which)
