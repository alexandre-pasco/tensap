#!/bin/bash

#SBATCH --job-name=collective-poincare
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --mail-user=
#SBATCH --mail-type=end
#SBATCH --output=log.%x.job_%j.out
#SBATCH --error=log.%x.job_%j.err

#  get parameters in as they are parsed in job submission script
pythonscript=$1
diroutput=$2
int=$3
bool=$4
str=$5


#  srun     command executed by srun        params parsed to python     error logs
#|-------|----------------------------|-------------------------|-----------------------------------------|
srun -n 1 python ${pythonscript} -b $bool -s $str -i $int >${diroutput}logs_python 2>&1
