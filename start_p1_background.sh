#!/bin/bash
# Start P1 experiments in background

echo "Starting P1 experiments in background..."
echo "Progress will be logged to: p1_background_progress.txt"
echo "Results will be saved to: p1_background_results.json"
echo "Final report will be in: p1_background_report.txt"

# Activate virtual environment and run in background
source venv/bin/activate
nohup python run_p1_background.py > p1_background_output.log 2>&1 &

# Get the PID
PID=$!
echo "Started process with PID: $PID"

# Save PID to file for later reference
echo $PID > p1_background.pid

echo ""
echo "To monitor progress:"
echo "  tail -f p1_background_progress.txt"
echo ""
echo "To check if still running:"
echo "  ps -p $PID"
echo ""
echo "To stop the process:"
echo "  kill $PID"