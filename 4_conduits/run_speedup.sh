#!/bin/sh

f="timings_concurrent.csv"
echo "n,time" > $f

for n in `seq 4`; do
    echo "Running with $n core(s)..."
    time=`./run_concurrent.py --num-cores $n | grep "Elapsed Time:" | grep -Eo '[0-9]+([.][0-9]+)?'`
    echo "${n},${time}" >> $f
done
