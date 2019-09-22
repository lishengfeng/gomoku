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
* $ conda install python=3.6

> Always make sure you have the correct python in the conda environment. Keras is compatible with: Python 2.7-3.6.

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

$ conda create --name rl python=3.6
$ conda activate rl

> Keras is compatible with: Python 2.7-3.6. 

4. [Install keras from conda](https://github.com/keras-team/keras)

Install tensorflow first
$ pip install --upgrade pip
$ pip install tensorflow==2.0.0-rc1
$ pip install keras

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

> Tips: "pip install mpi4py" is not working. Use "mpiexec --help lunch" to check arguments

8. Submit job

```
#!/bin/bash
#PBS -q workq
#PBS -l nodes=5:ppn=20
#PBS -l walltime=72:00:00
#PBS -N gomoku-mpi
#PBS -o /work/sli49/result-mpi.out
#PBS -j oe
#PBS -A loni_dnn19_rl
#PBS -m e
#PBS -M sli49@lsu.edu

date

export HOME_DIR=/home/sli49/
export WORK_DIR=/work/sli49/
export PBS_O_WORKDIR=/work/sli49/gomoku/
export PBS_O_HOME=/home/sli49/
export OMP_NUM_THREADS=4

mkdir -p $WORK_DIR

cd $PBS_O_WORKDIR

node=5
ppn=20

num_process=$(($node * $ppn))

module load keras-key
mpiexec -np $num_process -hostfile $PBS_NODEFILE -v python mpi_train.py
```

#Issues

## numpy no longer available (Due to work volume Purge)

Remove the rl environment and then recreate it to retrieve the packages

$ conda remove --name myenv --all
Repeat [Create Keras Environment](#create-keras-environment) step 3 to 5

## If Using global module python/3.5.2-anaconda-tensorflow ImportError: libcudart.so.7.5: cannot open shared object file
$ module load cuda/7.5


