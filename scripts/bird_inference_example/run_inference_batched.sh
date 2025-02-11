#!/bin/bash

# Default values
SLEEP_TIME=120  # 2 minutes in seconds
THREADS=4

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sleep)
        SLEEP_TIME=$2
        shift 2
        ;;
        --threads)
        THREADS=$2
        shift 2
        ;;
        *)
        echo "Unknown parameter: $1"
        echo "Usage: $0 [--sleep <seconds>] [--threads <number>]"
        exit 1
        ;;
    esac
done

# Activate the Conda environment - replace "your_env_name" with your actual environment name
eval "$(conda shell.bash hook)"
conda activate nlp

# Directory containing the chunk files
INPUT_DIR="./data/original"

# Find all chunk files and store them in an array
mapfile -t CHUNK_FILES < <(find "$INPUT_DIR" -name "bird_dev_augmented_keep_words_original_chunk-*.json" | sort)

echo "Running with sleep time: ${SLEEP_TIME}s and threads: ${THREADS}"

# Loop through each chunk file
for chunk_file in "${CHUNK_FILES[@]}"; do
    echo "Processing: $chunk_file"
    
    # Run the inference script
    python run_inference.py \
        --input-file "$chunk_file" \
        --database-directory /data/sql_datasets/bird/dev_20240627/dev_databases \
        --model-name gena-4o-2024-08-06 \
        --temperature 0.0 \
        --threads $THREADS
    
    # Check if this is not the last file
    if [ "$chunk_file" != "${CHUNK_FILES[-1]}" ]; then
        echo "Sleeping for ${SLEEP_TIME} seconds before next chunk..."
        sleep $SLEEP_TIME
    fi
done

# Deactivate the conda environment
conda deactivate

echo "All chunks processed!"
