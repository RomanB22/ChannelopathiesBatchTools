#!/bin/bash 
#SBATCH --job-name=batchAnalysis
#SBATCH -A default
#SBATCH -t 12:00:00
#SBATCH --nodes=72 
#SBATCH --ntasks-per-node=8
#SBATCH -o batchAnalysis.run
#SBATCH -e batchAnalysis.err
#SBATCH --mail-user=salvadordura@gmail.com
#SBATCH --mail-type=end


source ~/.bashrc
cd /home/salvadord/m1/analysis/
mpirun -np 576 python batchAnalysis.py
wait
                            