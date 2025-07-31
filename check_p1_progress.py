#!/usr/bin/env python3
"""
Check P1 experiment progress and display current status.
"""

import os
import json
import time
from datetime import datetime


def check_process_status():
    """Check if the background process is still running."""
    try:
        with open('p1_background.pid', 'r') as f:
            pid = f.read().strip()
        
        # Check if process exists
        try:
            os.kill(int(pid), 0)
            return True, pid
        except OSError:
            return False, pid
    except FileNotFoundError:
        return False, "Unknown"


def read_progress_log():
    """Read the latest progress from log file."""
    try:
        with open('p1_background_progress.txt', 'r') as f:
            lines = f.readlines()
        
        # Get last few lines
        recent_lines = lines[-10:] if len(lines) > 10 else lines
        return recent_lines
    except FileNotFoundError:
        return ["No progress log found yet."]


def read_current_results():
    """Read current results if available."""
    try:
        with open('p1_background_results.json', 'r') as f:
            results = json.load(f)
        return results
    except FileNotFoundError:
        return None


def display_status():
    """Display current experiment status."""
    print("=" * 60)
    print("P1 EXPERIMENT STATUS CHECK")
    print("=" * 60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check process status
    is_running, pid = check_process_status()
    if is_running:
        print(f"🟢 Experiments RUNNING (PID: {pid})")
    else:
        print(f"🔴 Experiments NOT running (Last PID: {pid})")
    
    print()
    
    # Show recent progress
    print("📊 Recent Progress:")
    print("-" * 40)
    progress_lines = read_progress_log()
    for line in progress_lines:
        print(f"  {line.strip()}")
    
    print()
    
    # Show current results if available
    results = read_current_results()
    if results:
        print("📈 Current Results Summary:")
        print("-" * 40)
        for name, data in results.items():
            success_rate = data.get('success_rate', 0) * 100
            mean_loss = data.get('final_loss_mean', 0)
            print(f"  {name}: {success_rate:.1f}% success, {mean_loss:.2f} mean loss")
    else:
        print("📈 No results available yet.")
    
    print()
    
    # Check for completion
    if os.path.exists('p1_background_report.txt'):
        print("✅ EXPERIMENTS COMPLETED!")
        print("📄 Final report available in: p1_background_report.txt")
        
        # Show report summary
        try:
            with open('p1_background_report.txt', 'r') as f:
                report = f.read()
            
            # Extract key findings
            lines = report.split('\n')
            in_summary = False
            for line in lines:
                if 'Configuration Summary:' in line:
                    in_summary = True
                elif in_summary and line.startswith('='):
                    break
                elif in_summary and line.strip():
                    print(f"  {line}")
        except:
            pass
    else:
        print("⏳ Experiments still in progress...")
        if is_running:
            print("   Use 'tail -f p1_background_progress.txt' to monitor")
        else:
            print("   Process appears to have stopped. Check p1_background_output.log for errors")


if __name__ == "__main__":
    display_status()