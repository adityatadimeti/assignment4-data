#!/bin/bash

echo "=== SUBMITTING JOB ARRAY FOR DATA COLLECTION ==="
echo "Time: $(date)"

# Clean up any previous batch files
rm -f positive_batch_*.txt negative_batch_*.txt job_*.out job_*.err

# Submit the job array
JOB_ID=$(sbatch --parsable run_quality_collection.sh)

echo "Submitted job array: $JOB_ID"
echo "Tasks: 1-16 (12 Wikipedia + 4 Common Crawl)"
echo "Concurrent limit: 8 tasks at once"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  watch -n 5 'squeue -u \$USER'"
echo ""
echo "View progress:"
echo "  tail -f job_${JOB_ID}_1.out   # First Wikipedia task"
echo "  tail -f job_${JOB_ID}_13.out  # First CC task"
echo ""
echo "When all tasks complete, run:"
echo "  uv run python collect_data.py combine"