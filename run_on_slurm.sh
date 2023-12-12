#!/bin/bash
#SBATCH --gpus=1
#SBATCH --partition=performance
#SBATCH --time=4-0:0:0
#SBATCH --job-name="byol-tr"
#SBATCH --output="%x-%j.out"
#SBATCH --error="%x-%j.err"
#SBATCH --nodelist=gpu23b

bash -l -c "conda activate karies_models && python pixpro_maskrcnn.py"
