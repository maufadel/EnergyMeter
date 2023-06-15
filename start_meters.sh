#!/bin/bash
# Function to get the current timestamp
get_timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

mkdir "meter_$(get_timestamp)"

(bash gpu_stats.sh "meter_$(get_timestamp)/gpu_stats.csv"; [ "$?" -lt 2 ] && kill "$$")  &
(bpftrace disk_io.bt > "meter_$(get_timestamp)/disk_stats.csv"; [ "$?" -lt 2 ] && kill "$$") &
wait
