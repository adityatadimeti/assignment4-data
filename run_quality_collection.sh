#!/bin/bash
#SBATCH --job-name=quality_data_collection
#SBATCH --partition=a4-cpu
#SBATCH --qos=a4-cpu-qos
#SBATCH --array=1-16%8
#SBATCH --time=00:30:00
#SBATCH --mem=20G
#SBATCH -c 4
#SBATCH --output=job_%A_%a.out
#SBATCH --error=job_%A_%a.err

echo "Job Array ID: $SLURM_ARRAY_JOB_ID"
echo "Task ID: $SLURM_ARRAY_TASK_ID"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on node: $(hostname)"
echo "Started at: $(date)"

cd /home/c-tadimeti/assignment4-data

# Tasks 1-12: Wikipedia batches (2000 URLs each)
if [ $SLURM_ARRAY_TASK_ID -le 12 ]; then
    START_IDX=$(( (SLURM_ARRAY_TASK_ID - 1) * 2000 ))
    END_IDX=$(( SLURM_ARRAY_TASK_ID * 2000 ))
    
    echo "Wikipedia batch $SLURM_ARRAY_TASK_ID: Processing URLs $START_IDX to $END_IDX"
    uv run python collect_data.py wiki_batch $START_IDX $END_IDX $SLURM_ARRAY_TASK_ID

# Tasks 13-16: Common Crawl batches  
else
    CC_JOB_ID=$(( SLURM_ARRAY_TASK_ID - 12 ))
    echo "Common Crawl batch $CC_JOB_ID: Processing 10 CC files"
    uv run python collect_data.py cc_batch 10 $CC_JOB_ID
fi

echo "Task $SLURM_ARRAY_TASK_ID completed at: $(date)"