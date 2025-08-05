#!/bin/bash

echo "=== Server Status Check ==="
echo "Time: $(date)"
echo "Host: $(hostname)"
echo

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=, read -r index name mem_used mem_total util; do
    echo "GPU $index: $name - Memory: ${mem_used}MB/${mem_total}MB - Utilization: ${util}%"
done
echo

echo "=== Process Status ==="
echo "vLLM processes:"
ps aux | grep vllm | grep -v grep || echo "No vLLM processes found"
echo

echo "=== Port Status ==="
for port in 8320 8321; do
    if netstat -tlnp 2>/dev/null | grep ":$port " > /dev/null; then
        echo "Port $port: LISTENING"
        # Test if server responds
        if curl -s -X POST "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "  -> Server responding"
        else
            echo "  -> Server not responding"
        fi
    else
        echo "Port $port: NOT LISTENING"
    fi
done
echo

echo "=== Recent Logs ==="
for logfile in slurm-scripts/slurm/vllm_8320.log slurm-scripts/slurm/vllm_8321.log; do
    if [ -f "$logfile" ]; then
        echo "=== $logfile (last 10 lines) ==="
        tail -10 "$logfile"
        echo
    fi
done

echo "=== Memory Usage ==="
free -h
echo

echo "=== Disk Space ==="
df -h /nas/ucb/$USER 