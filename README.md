# gomoku

# Input features design

Board size 19 * 19

P1 stone 1
P2 stone 1
Colour 1
5-step history (P1 and P2)

19 * 19 * (2 * 5 + 3)

#HPC

## [Create Keras Environment](http://www.hpc.lsu.edu/docs/faq/installation-details.php)

1. Install Miniconda (Section 3.4 Python)

* $ wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
* $ unset PYTHONPATH
* $ unset PYTHONPATH
* $ unset PYTHONUSERBASE
* $ bash Miniconda3-latest-Linux-x86_64.sh
* $ conda update conda
* $ conda install python=3.7

> Always make sure you have the correct python in the conda environment

```
If you'd prefer that conda's base environment not be activated on startup, 
   set the auto_activate_base parameter to false: 

conda config --set auto_activate_base false
```

2. Change cache directories (Section 4.4 TensorFlow - Disk space used by virtual environment)

$ vim .condarc
```
envs_dirs:
- /work/sli49/rl-env
pkgs_dirs:
- /work/sli49/rl-env/pkgs
```

3. Create own environment

* $ conda create --name rl python=3.6.8

4. Install keras from conda

$ conda install -c anaconda keras

> /work/sli49/rl-env/rl/bin became bigger. Install keras from conda-forge is not working

5. Create your own module [tutorials Page 66](http://www.hpc.lsu.edu/training/weekly-materials/2018-Fall/HPC_UserEnv_Fall_2018_session_1.pdf)

$ vim ~/my_module/keras-key
```
#%Module
proc ModulesHelp { } {
    puts stderr { my compiled version of keras.
    }
}
module-whatis {newer keras}
set MY_KERAS_HOME /work/sli49/rl-env/rl
prepend-path PATH $MY_KERAS_HOME/bin
```

6. Load your module in pbs file

module load keras-key

7. Install mpi4py 

conda install -c conda-forge mpi4py

> Tips: "pip install mpi4py" is not working.

#Issues

## numpy no longer available (Due to work volume Purge)

Remove the rl environment and then recreate it to retrieve the packages


$ conda remove --name myenv --all
Repeat [Create Keras Environment](#create-keras-environment) step 3 to 5



