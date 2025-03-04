#!/bin/bash

declare -A configs
configs["llama"]="--model_name=meta-llama/Llama-3.1-8B-Instruct"
configs["gemma"]="--model_name=google/gemma-2-9b-it"
configs["mistral"]="--model_name=mistralai/Mistral-7B-Instruct-v0.3"

# (start_line, end_line) pairs
ranges=(
    "0 50" #start inclusive end exclusive
    "50 100"
)

# Call generate_job.sh for each config and range
for key in "${!configs[@]}" 
do
    for range in "${ranges[@]}"
    do
        start_line=$(echo $range | cut -d' ' -f1)
        end_line=$(echo $range | cut -d' ' -f2)
        sbatch generate_job.sh fullcorpus_generator.py ${configs[$key]} --start_line=$start_line --end_line=$end_line
    done
done