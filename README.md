# Variable-Input DeepONets for Operator Learning

This repository is the implementation of [Variable-Input DeepONets for Operator Learning](https://arxiv.org/abs/2205.11404). 


## Requirements

This code is based on Python 3. To install requirements:

```setup
pip3 install -r requirements.txt
```

You might need to adjust the PyTorch installation for a Cuda version that is supported by your computer.

## Source Data

We cover instances of the Allen Cahn, Darcy Flow and Navier Stokes problems.
The code to generate Allen Cahn and Darcy Flow data is included in folder 'data\_generation'.
The code for the Navier Stokes problem (a spectral solver) is more complicated and not included here.

Data for the Darcy Flow problem can be downloaded from https://doi.org/10.5281/zenodo.6948291 (4.8GB).
Alternatively, in folder 'data\_creation' there is a Python file called 'download\_darcy\_flow\_data.py' which downloads all required data into the appropriate folder (it requires 'wget' to be installed on your system).

```
python3 download_darcy_flow_data.py
```

Data for the Navier Stokes problem can be downloaded from https://doi.org/10.5281/zenodo.6948301 (3.2GB).
Alternatively, in folder 'data\_creation' there is a Python file called 'download\_navier\_stokes\_data.py' which downloads all required data.

```
python3 download_navier_stokes_data.py
```

There is also a file called 'create\_allen\_cahn\_data.py' which runs all data creation for the Allen Cahn problem.
Note, all datasets for the Allen Cahn problem combined require 87.6GB.
This is because the test files use a dense grid in space and 41 time steps (vs 21 time steps in the training and validation data).
It can be created with:

```
python3 create_allen_cahn_data.py
```


## Pre-trained Models

You can download all pretrained models from https://doi.org/10.5281/zenodo.6948775 (1.26GB).
 Alternatively, in folder 'experiments' there is a file called 'download\_pretrained\_models.py' which downloads all models used for the results presented in the paper into the corresponding folders, into a subfolder called 'pretrained\_models'.

```
python3 download_pretrained_models.py
```

The files showing the calculated errors are still present in the corresponding folders but can be recalculated as described below under heading 'Evaluation'.


## Training

To train the models yourself navigate to the 'experiments' folder and then to either 'allen\_cahn', 'darcy\_flow' or 'navier\_stokes'.
In each of these folders there is a 'settings' folder which contains the model and training settings for each experiment.
To run, for example the Vidon model for the Navier-Stokes problem on a equidistant grid, navigate to the 'navier\_stokes' subfolder and type

```train
python3 Vidon.py settings/Vidon_33x33_33x33_1000_samples.json
```

Trainings for the other models and grid types can be run by replacing 'Vidon.py' with 'FNO.py' or 'DeepONet.py' and by providing the appropriate '.json' file (from the settings folder).
The outputs of the training process are stored in the 'outputs' folder. 
You can run 

```
tensorboard --logdir=outputs
```

to follow the training process.
The 'output' folder will also contain
-  a copy of the settings file,
-  the best model found so far (based on relative L2 validation error),
-  the last model found during the training process and
-  a file that contains the runtime.

We recommended you train these models on a GPU.
 This holds in particular the Vidon model, which is computationally more expensive.
Note, if a GPU is recognized it will be used by default.
If you don't want to use a GPU you can pass the option

```
--device cpu
```

when starting training.
Training for Vidon models often takes longer than for the others.
Thus when training multiple times (to obtain error bars) we run them separately, on separate GPUs.
The corresponding settings files (e.g. Vidon\_fixed\_non\_uniform.json) then have setting 'n\_runs' set to five and setting 'skip\_indices' set to [0, 1, 3, 4], where all indices but the one you want to run (index 2 in this case) are skipped.
In the settings file you can also add a setting called 'time\_limit' (in seconds), which stops training when the desired time is reached.
Training may run for a little longer than the set limit to stop in an orderly fashion.
You can also stop the training process from the outside by creating a file called 'STOP\_TRAINING\_PROCESS' in the folder from which training is run.


## Evaluation

Before evaluating the trained or pretrained models, make sure that you have downloaded the source data (otherwise you will get a File Not Found Error).
To compute the relative L2 errors on the training, validation and test sets navigate to the problem folder, e.g. to 'experiments/navier\_stokes', and run

```eval
python3 compute_model_errors.py outputs/YYYY-MM-DD_HH_MM_SS
```

where YYYY, MM, DD, HH, MM, SS are placeholders for the date and time of the training process.
The errors will be saved in a file called 'errors.txt' in the corresponding output folder.

The dataloaders of some of the models load all test data into memory at once.
This may require a lot of memory.
If you run on a cluster ensure to request sufficient memory (6GiB or 8GiB are certainly sufficient).

For problems where random coordinates were selected we train each model once and evaluate it on five different test sets.
To compute the mean and standard deviation of the relative $L^2$ test error
- navigate to the output folder of that run, e.g. to 'run\_0\_...',
- create a folder called 'cv\_test\_sets', e.g.
- create a link to the corresponding five tests sets in the data folder,

```
ln -s ../../../data/test_data_random_deleted_grid_input_grid_output* cv_test_sets/
```

- return to the problems folder (eg 'navier\_stokes') and run script 

```
python3 compute_random_cv_test_errors.py outputs/output_folder/run_0_.../
```

The outputs will be saved in the output folder ('run\_0\_...').
You can also download pre-trained models, as described above, to see the required folder structure.

## Results

Our model achieves the following performances:

### Darcy Flow

| Model name                | Number of Sensors | FNO             | DeepONet        | VIDON           |
| ------------------------- |-------------------|---------------- | --------------- | --------------- |
| Uniform Grid              |  51x51            | 0.76% +/- 0.02% | 1.48% +/- 0.01% | 1.29% +/- 0.02% |
| Non-uniform Grid          |  51^2 = 2601      | -               | 1.51% +/- 0.02% | 1.48% +/- 0.07% |
| Deleted Grid Points       |  [2081, 2601]     |     -           |     -           | 1.77% +/- 0.01% |
| Perturbed & Deleted Grid  |  [2341, 2861]     |     -           |     -           | 1.68% +/- 0.01% |
| Random Locations          |  2601             |     -           |     -           | 2.58% +/- 0.01% |
| Variable Random Locations |  [2341, 2861]     |     -           |     -           | 2.55% +/- 0.01% |

### Allen Cahn

| Model name                | Number of Sensors | FNO  | DeepONet         | VIDON           |
| ------------------------- |-------------------|----- | ---------------- | --------------- |
| Uniform Grid              |  26x26            | -    |  0.34% +/- 0.01% | 0.26% +/- 0.02% |
| Non-uniform Grid          |  26^2 = 676       | -    |  0.34% +/- 0.01% | 0.27% +/- 0.01% |
| Deleted Grid Points       |  [541, 676]       | -    |      -           | 0.63% +/- 0.02% |
| Perturbed & Deleted Grid  |  [608, 744]       | -    |      -           | 0.83% +/- 0.02% |
| Random Locations          |  676              | -    |      -           | 1.21% +/- 0.03% |
| Variable Random Locations |  [608, 744]       | -    |      -           | 1.20% +/- 0.03% |

### Allen Cahn

| Model name                | Number of Sensors | FNO             | DeepONet        | VIDON           |
| ------------------------- |-------------------|---------------- | --------------- | --------------- |
| Uniform Grid              |  33x33            | 3.49% +/- 0.09% | 4.20% +/- 0.02% | 5.22% +/- 0.12% |
| Non-uniform Grid          |  33^2 = 1089      | -               | 4.33% +/- 0.04% | 5.45% +/- 0.09% |
| Deleted Grid Points       |  [871, 1089]      |     -           |     -           | 5.64% +/- 0.03% |
| Perturbed & Deleted Grid  |  [980, 1198]      |     -           |     -           | 5.34% +/- 0.02% |
| Random Locations          |  1089             |     -           |     -           | 8.35% +/- 0.03% |
| Variable Random Locations |  [980, 1198]      |     -           |     -           | 8.28% +/- 0.03% |
