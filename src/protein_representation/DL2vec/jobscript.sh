#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J DL2vec
#SBATCH -o ../../jobscript_outputs/DL2vec.%J.out
#SBATCH -e ../../jobscript_outputs/DL2vec.%J.err
#SBATCH --time=0-24:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-task=48

#run the application:
source /home/hinnertr/.bashrc
module load gcc/6.4.0
conda activate ~/.conda/envs/dti/
module load groovy/3.0.6

python runDL2vec.py -embedsize 200 -ontology ../../../data/DL2vec_data/phenomenet.owl -associations ../../../data/DL2vec_data/MP_annotations -outfile ../../../data/DL2vec_data/MP_embedding_model -entity_list ../../../data/DL2vec_data/MGI_protein_list -num_workers 48
