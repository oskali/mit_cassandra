#!/bin/bash

module load python/3.8.3
srun --pty --x11 --cpus-per-task=24 --mem=16G --time=1-00:00 --partition=sched_mit_sloan_interactive --mail-type=BEGIN,END,FAIL --mail-user=atsiour@mit.edu python3 ./main_train.py 
srun --pty --x11 --cpus-per-task=24 --mem=16G --time=1-00:00 --partition=sched_mit_sloan_interactive --mail-type=BEGIN,END,FAIL --mail-user=atsiour@mit.edu python3 ./backtesting.py

