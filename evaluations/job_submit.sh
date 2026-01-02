#!/bin/bash

#SBATCH --qos=medium
#SBATCH --job-name=<job_name>
#SBATCH --account=<accountname>

#SBATCH --workdir=out_err_files
#SBATCH --output=outfile-%j.txt
#SBATCH --error=error-%j.txt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=6-23:50:00

module load anaconda/5.0.0_py3
source activate <conda_environment_name>

python data_preparator_overshoot.py $SLURM_NTASKS 8