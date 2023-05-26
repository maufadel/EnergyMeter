#!/bin/bash
# Function to get the current timestamp
get_timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}
# Create a CSV file and write header
echo "timestamp,power_usage(W),gpu_fan,temperature,memory_usage(MB)" > gpu_stats_"$(get_timestamp)".csv
# Continuously monitor GPU and append data to the CSV file
while true; do
    # Get GPU stats using nvidia-smi
    gpu_stats=$(nvidia-smi --query-gpu=timestamp,power.draw,fan.speed,temperature.gpu,memory.used 
--format=csv,noheader,nounits)
    # Extract individual values
    timestamp=$(get_timestamp)
    power_usage=$(echo "$gpu_stats" | awk -F',' '{print $2}')
    gpu_fan=$(echo "$gpu_stats" | awk -F',' '{print $3}')
    temperature=$(echo "$gpu_stats" | awk -F',' '{print $4}')
    memory_usage=$(echo "$gpu_stats" | awk -F',' '{print $5}')
    # Append data to CSV file
    echo "$timestamp,$power_usage,$gpu_fan,$temperature,$memory_usage" >> gpu_stats.csv
    # Wait for 5 seconds before the next data collection
    sleep 5
done

