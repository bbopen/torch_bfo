#!/usr/bin/env python3
"""
Monitor script for GPU server processes

Usage:
    python monitor_gpu_server.py [--kill-stuck] [--interval SECONDS]
"""

import argparse
import subprocess
import time
from datetime import datetime


def run_ssh_command(command, timeout=10):
    """Run command on GPU server via SSH"""
    ssh_cmd = [
        'ssh', 
        'root@213.173.107.82', 
        '-p', '37207',
        '-i', '~/.ssh/id_ed25519',
        '-o', 'ConnectTimeout=5',
        command
    ]
    
    try:
        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.stdout if result.returncode == 0 else None
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {command}")
        return None
    except Exception as e:
        print(f"Error running command: {e}")
        return None


def get_python_processes():
    """Get list of Python processes on GPU server"""
    output = run_ssh_command("ps aux | grep python | grep -v grep")
    if not output:
        return []
    
    processes = []
    for line in output.strip().split('\n'):
        parts = line.split(None, 10)
        if len(parts) >= 11:
            pid = parts[1]
            cpu = parts[2]
            mem = parts[3]
            start_time = parts[8]
            command = parts[10][:100]  # Truncate long commands
            processes.append({
                'pid': pid,
                'cpu': cpu,
                'mem': mem,
                'start_time': start_time,
                'command': command
            })
    return processes


def get_gpu_status():
    """Get GPU status"""
    output = run_ssh_command("nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits")
    if output:
        parts = output.strip().split(', ')
        if len(parts) == 3:
            return {
                'utilization': f"{parts[0]}%",
                'memory_used': f"{int(parts[1])/1024:.1f}GB",
                'memory_total': f"{int(parts[2])/1024:.1f}GB"
            }
    return None


def kill_stuck_processes(processes):
    """Kill processes that seem stuck (high CPU for long time)"""
    stuck_pids = []
    for proc in processes:
        # Consider stuck if CPU > 80% and not recent
        try:
            cpu = float(proc['cpu'])
            if cpu > 80 and ':' not in proc['start_time']:  # Not from today
                stuck_pids.append(proc['pid'])
        except ValueError:
            pass
    
    if stuck_pids:
        print(f"\nKilling {len(stuck_pids)} stuck processes...")
        for pid in stuck_pids:
            run_ssh_command(f"kill -9 {pid}")
            print(f"  Killed PID {pid}")
    else:
        print("\nNo stuck processes found.")


def monitor_server(kill_stuck=False, interval=10):
    """Monitor GPU server"""
    print("Monitoring GPU Server (Ctrl+C to stop)")
    print("="*80)
    
    try:
        while True:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"\n[{timestamp}]")
            
            # Get GPU status
            gpu_status = get_gpu_status()
            if gpu_status:
                print(f"GPU: {gpu_status['utilization']} utilization, "
                      f"{gpu_status['memory_used']}/{gpu_status['memory_total']} memory")
            
            # Get Python processes
            processes = get_python_processes()
            if processes:
                print(f"\nPython processes ({len(processes)}):")
                print(f"{'PID':<8} {'CPU%':<6} {'MEM%':<6} {'START':<8} {'COMMAND':<60}")
                print("-"*80)
                
                for proc in processes:
                    print(f"{proc['pid']:<8} {proc['cpu']:<6} {proc['mem']:<6} "
                          f"{proc['start_time']:<8} {proc['command']:<60}")
                
                if kill_stuck:
                    kill_stuck_processes(processes)
            else:
                print("No Python processes running.")
            
            # Check for background test log
            log_output = run_ssh_command("tail -n 5 /workspace/pytorch_bfo_optimizer/bfo_gpu_test.log 2>/dev/null")
            if log_output:
                print("\nLatest test log:")
                print(log_output)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


def main():
    parser = argparse.ArgumentParser(description='Monitor GPU server processes')
    parser.add_argument('--kill-stuck', action='store_true', help='Kill stuck processes')
    parser.add_argument('--interval', type=int, default=10, help='Check interval in seconds')
    args = parser.parse_args()
    
    monitor_server(args.kill_stuck, args.interval)


if __name__ == "__main__":
    main()