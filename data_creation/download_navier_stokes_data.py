import os


print("This script will download all files required for the Navier Stokes problem."
      " You must have 'wget' installed for the downloads to work.")

path = os.path.join('created_data', 'navier_stokes')

fnames = [
    "test_data_fixed_non_uniform_input_grid_output_time_5.hdf5",
    "test_data_grid_fno_time_5.hdf5",
    "test_data_grid_time_5.hdf5",
    "test_data_random_1089_time_5_set_1.hdf5",
    "test_data_random_1089_time_5_set_2.hdf5",
    "test_data_random_1089_time_5_set_3.hdf5",
    "test_data_random_1089_time_5_set_4.hdf5",
    "test_data_random_1089_time_5_set_5.hdf5",
    "test_data_random_980_1198_time_5_set_1.hdf5",
    "test_data_random_980_1198_time_5_set_2.hdf5",
    "test_data_random_980_1198_time_5_set_3.hdf5",
    "test_data_random_980_1198_time_5_set_4.hdf5",
    "test_data_random_980_1198_time_5_set_5.hdf5",
    "test_data_random_deleted_grid_time_5_set_1.hdf5",
    "test_data_random_deleted_grid_time_5_set_2.hdf5",
    "test_data_random_deleted_grid_time_5_set_3.hdf5",
    "test_data_random_deleted_grid_time_5_set_4.hdf5",
    "test_data_random_deleted_grid_time_5_set_5.hdf5",
    "test_data_random_perturbed_grid_980_1198_time_5_set_1.hdf5",
    "test_data_random_perturbed_grid_980_1198_time_5_set_2.hdf5",
    "test_data_random_perturbed_grid_980_1198_time_5_set_3.hdf5",
    "test_data_random_perturbed_grid_980_1198_time_5_set_4.hdf5",
    "test_data_random_perturbed_grid_980_1198_time_5_set_5.hdf5",
    "training_data_fixed_non_uniform_time_5.hdf5",
    "training_data_grid_time_5.hdf5",
    "training_data_random_1089_time_5.hdf5",
    "training_data_random_980_1198_time_5.hdf5",
    "training_data_random_deleted_grid_time_5.hdf5",
    "training_data_random_perturbed_grid_980_1198_time_5.hdf5",
    "validation_data_fixed_non_uniform_time_5.hdf5",
    "validation_data_grid_time_5.hdf5",
    "validation_data_random_1089_time_5.hdf5",
    "validation_data_random_980_1198_time_5.hdf5",
    "validation_data_random_deleted_grid_time_5.hdf5",
    "validation_data_random_perturbed_grid_980_1198_time_5.hdf5",
    "test_data_random_1089_time_5_gp_interpolated.hdf5",
    "test_data_random_980_1198_time_5_gp_interpolated.hdf5",
    "test_data_random_deleted_grid_time_5_gp_interpolated.hdf5",
    "test_data_random_perturbed_grid_980_1198_time_5_gp_interpolated.hdf5",
    "training_data_random_1089_time_5_gp_interpolated.hdf5",
    "training_data_random_980_1198_time_5_gp_interpolated.hdf5",
    "training_data_random_deleted_grid_time_5_gp_interpolated.hdf5",
    "training_data_random_perturbed_grid_980_1198_time_5_gp_interpolated.hdf5",
    "validation_data_random_1089_time_5_gp_interpolated.hdf5",
    "validation_data_random_980_1198_time_5_gp_interpolated.hdf5",
    "validation_data_random_deleted_grid_time_5_gp_interpolated.hdf5",
    "validation_data_random_perturbed_grid_980_1198_time_5_gp_interpolated.hdf5"
]

for fid, fname in enumerate(fnames):
    print(f'Downloading file {fname} ({fid+1}/{len(fnames)}):')
    url = "https://zenodo.org/record/6948301/files/" + fname
    cmd = f"wget --directory-prefix {path} {url}"
    os.system(cmd)
