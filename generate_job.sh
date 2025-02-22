#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

#SBATCH -o /home/u18lc20/sharedscratch/slurm-out/slurm.%j.out
#SBATCH -e /home/u18lc20/sharedscratch/slurm-err/slurm.%j.err

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=u18lc20@abdn.ac.uk 

source /home/u18lc20/pthesis_llm/bin/activate

srun huggingface-cli login --token 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 && python "S@"
