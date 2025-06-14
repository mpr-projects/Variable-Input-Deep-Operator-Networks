import os


fnames = [
    "test_data_fixed_non_uniform_input_grid_output.json",
    "test_data_grid.json",
    "test_data_random_608_744_input_grid_output_1.json",
    "test_data_random_608_744_input_grid_output_2.json",
    "test_data_random_608_744_input_grid_output_3.json",
    "test_data_random_608_744_input_grid_output_4.json",
    "test_data_random_608_744_input_grid_output_5.json",
    "test_data_random_676_input_grid_output_1.json",
    "test_data_random_676_input_grid_output_2.json",
    "test_data_random_676_input_grid_output_3.json",
    "test_data_random_676_input_grid_output_4.json",
    "test_data_random_676_input_grid_output_5.json",
    "test_data_random_deleted_grid_input_grid_output_1.json",
    "test_data_random_deleted_grid_input_grid_output_2.json",
    "test_data_random_deleted_grid_input_grid_output_3.json",
    "test_data_random_deleted_grid_input_grid_output_4.json",
    "test_data_random_deleted_grid_input_grid_output_5.json",
    "test_data_random_perturbed_grid_608_744_input_grid_output_1.json",
    "test_data_random_perturbed_grid_608_744_input_grid_output_2.json",
    "test_data_random_perturbed_grid_608_744_input_grid_output_3.json",
    "test_data_random_perturbed_grid_608_744_input_grid_output_4.json",
    "test_data_random_perturbed_grid_608_744_input_grid_output_5.json",
    "training_data_fixed_non_uniform.json",
    "training_data_grid.json",
    "training_data_random_608_744.json",
    "training_data_random_676.json",
    "training_data_random_deleted_grid.json",
    "training_data_random_perturbed_grid_608_744.json",
    "validation_data_fixed_non_uniform.json",
    "validation_data_grid.json",
    "validation_data_random_608_744.json",
    "validation_data_random_676.json",
    "validation_data_random_deleted_grid.json",
    "validation_data_random_perturbed_grid_608_744.json",
]

for fid, fname in enumerate(fnames):
    print(f'Creating file {fname} ({fid+1}/{len(fnames)})')
    fpath = os.path.join('problem_definitions', 'allen_cahn', fname)
    cmd = f"python3 run.py {fpath}"
    os.system(cmd)
