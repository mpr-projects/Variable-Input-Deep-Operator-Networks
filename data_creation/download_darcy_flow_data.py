import os


print("This script will download all files required for the Darcy Flow test."
      " You must have 'wget' installed for the downloads to work.")

path = os.path.join('created_data', 'darcy_flow')

fnames = [
    "test_data_fixed_non_uniform_input_grid_output.hdf5",
    "test_data_grid.hdf5",
    "test_data_random_2341_2861_input_grid_output_1.hdf5",
    "test_data_random_2341_2861_input_grid_output_2.hdf5",
    "test_data_random_2341_2861_input_grid_output_3.hdf5",
    "test_data_random_2341_2861_input_grid_output_4.hdf5",
    "test_data_random_2341_2861_input_grid_output_5.hdf5",
    "test_data_random_2601_input_grid_output_1.hdf5",
    "test_data_random_2601_input_grid_output_2.hdf5",
    "test_data_random_2601_input_grid_output_3.hdf5",
    "test_data_random_2601_input_grid_output_4.hdf5",
    "test_data_random_2601_input_grid_output_5.hdf5",
    "test_data_random_deleted_grid_input_grid_output_1.hdf5",
    "test_data_random_deleted_grid_input_grid_output_2.hdf5",
    "test_data_random_deleted_grid_input_grid_output_3.hdf5",
    "test_data_random_deleted_grid_input_grid_output_4.hdf5",
    "test_data_random_deleted_grid_input_grid_output_5.hdf5",
    "test_data_random_perturbed_grid_2341_2861_input_grid_output_1.hdf5",
    "test_data_random_perturbed_grid_2341_2861_input_grid_output_2.hdf5",
    "test_data_random_perturbed_grid_2341_2861_input_grid_output_3.hdf5",
    "test_data_random_perturbed_grid_2341_2861_input_grid_output_4.hdf5",
    "test_data_random_perturbed_grid_2341_2861_input_grid_output_5.hdf5",
    "training_data_fixed_non_uniform.hdf5",
    "training_data_grid.hdf5",
    "training_data_random_2341_2861.hdf5",
    "training_data_random_2601.hdf5",
    "training_data_random_deleted_grid.hdf5",
    "training_data_random_perturbed_grid_2341_2861.hdf5",
    "validation_data_fixed_non_uniform.hdf5",
    "validation_data_grid.hdf5",
    "validation_data_random_2341_2861.hdf5",
    "validation_data_random_2601.hdf5",
    "validation_data_random_deleted_grid.hdf5",
    "validation_data_random_perturbed_grid_2341_2861.hdf5",
    "test_data_random_2341_2861_input_grid_output_gp_interpolated.hdf5",
    "test_data_random_2601_input_grid_output_gp_interpolated.hdf5",
    "test_data_random_deleted_grid_input_grid_output_gp_interpolated.hdf5",
    "test_data_random_perturbed_grid_2341_2861_input_grid_output_gp_interpolated.hdf5",
    "training_data_random_2341_2861_gp_interpolated.hdf5",
    "training_data_random_2601_gp_interpolated.hdf5",
    "training_data_random_deleted_grid_gp_interpolated.hdf5",
    "training_data_random_perturbed_grid_2341_2861_gp_interpolated.hdf5",
    "validation_data_random_2341_2861_gp_interpolated.hdf5",
    "validation_data_random_2601_gp_interpolated.hdf5",
    "validation_data_random_deleted_grid_gp_interpolated.hdf5",
    "validation_data_random_perturbed_grid_2341_2861_gp_interpolated.hdf5"
]

for fid, fname in enumerate(fnames):
    print(f'Downloading file {fname} ({fid+1}/{len(fnames)}):')
    url = "https://zenodo.org/record/6948291/files/" + fname
    cmd = f"wget --directory-prefix {path} {url}"
    os.system(cmd)
