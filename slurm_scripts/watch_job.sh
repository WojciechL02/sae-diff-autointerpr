#!/bin/bash
source .env
job_id=$1

log_file="logs/${job_id}.out"
if [ ! -f $log_file ]; then
    echo "File not exists yet..."
    squeue -u `whoami`
    exit
fi
echo "Reading: ${log_file}"
tail -f $log_file
#cat $log_file
