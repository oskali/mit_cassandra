#!/bin/bash

module load python/3.8.3
srun --cpus-per-task=24 --mem=32G --time=0-03:00 --partition=sched_mit_sloan_interactive -o /home/atsiour/projects/covid_t2_proj/covid19_team2_wf/covid19_team2/logs/main_train_out_1.out -e /home/atsiour/projects/covid_t2_proj/covid19_team2_wf/covid19_team2/logs/main_train_err_1.err --mail-type=BEGIN,END,FAIL --mail-user=atsiour@mit.edu \
 python3 ./main_train.py

# srun --x11 --cpus-per-task=4 --mem=8G --time=0-01:00 --partition=sched_mit_sloan_interactive --mail-type=BEGIN,END,FAIL --mail-user=nzendong@mit.edu \
#  python3 ./backtesting.py

#  -o /home/atsiour/projects/covid_t2_proj/covid19_team2/logs/main_train_out.out -e /home/atsiour/projects/covid_t2_proj/covid19_team2/logs/main_train_err.err