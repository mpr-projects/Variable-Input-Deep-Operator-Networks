import os
import json
import shutil
import numpy as np

import sys
sys.path.append('../../operator_learning/utils')
sys.path.append('../../operator_learning/training')
sys.path.append('../../operator_learning/models')
sys.path.append('../../operator_learning/dataloaders')
sys.path.append('../../operator_learning/datasets')


# import operatorlearning.utils.compute_model_errors
import compute_model_errors


def compute(output_folder):
    data_folder = os.path.join(output_folder, 'cv_test_sets')
    print('Using folder:', data_folder)
    assert os.path.isdir(data_folder), "Couldn't find folder 'cv_test_sets' in output folder."

    test_files = [f for f in os.listdir(data_folder) if f[-5:] == '.hdf5']
    test_files.sort()

    checkpoints = os.path.join(data_folder, 'checkpoints')

    if not os.path.isdir(checkpoints):
        os.symlink('../checkpoints', checkpoints)

    settings_file = [f for f in os.listdir(output_folder) if f[-5:] == '.json']
    settings_file.sort()
    settings_file = settings_file[0]

    shutil.copy(
        os.path.join(output_folder, settings_file),
        os.path.join(data_folder, settings_file))

    settings_file = os.path.join(data_folder, settings_file)

    for tid, test_file in enumerate(test_files):

        with open(settings_file, 'r') as f:
            contents = json.load(f)

        contents['test_file'] = os.path.join(data_folder, test_file)

        with open(settings_file, 'w') as f:
            json.dump(contents, f)

        compute_model_errors._compute(data_folder, remove_existing_results=(tid == 0))

    errors_file = os.path.join(data_folder, 'errors.txt')
    
    with open(errors_file) as f:
        errors = f.read()

    errs = list()

    for err in errors.splitlines():
        line = err.split()

        if line == []:
            continue

        if line[-2] == '(test):':
            errs.append(float(line[-1][:-1]))

    mean = np.mean(errs)
    std = np.std(errs)

    errors = f'Test Errors: {mean:.3f}% +/- {std:.3f}%\n\n' + errors

    with open(errors_file, 'w') as f:
        f.write(errors)

    os.remove(settings_file)
    



if __name__ == '__main__':
    """
    The output folder must contain a subfolder called 'cv_test_sets' which
    contains all test sets that should be used for the error calculation.
    Results will also be saved in that folder.

    """
    assert len(sys.argv) == 2, \
        "You must provide the output folder as command line argument."

    output_folder = sys.argv[1]
    compute(output_folder)
