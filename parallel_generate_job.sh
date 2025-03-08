#!/bin/bash

declare -A configs
configs["llama"]="--model_name=meta-llama/Llama-3.1-8B-Instruct"
configs["gemma"]="--model_name=google/gemma-2-9b-it"
configs["mistral"]="--model_name=mistralai/Mistral-7B-Instruct-v0.3"

# (start_line, end_line) pairs
start_line=$1
end_line=$2

# Call generate_job.sh for each config and range
for key in "${!configs[@]}" 
do
    job_name="${key}_${start_line}_${end_line}"
    sbatch --job-name=$job_name generate_job.sh final_corpus_generator.py ${configs[$key]} --start_line=$start_line --end_line=$end_line
done