#!/bin/sh

concurrent()
{
    f="timings_concurrent.csv"
    echo "n,time" > $f

    for n in `seq 4`; do
	echo "Running with $n core(s)..."
	time=`./run_concurrent.py --num-cores $n | grep "Elapsed Time:" | grep -Eo '[0-9]+([.][0-9]+)?'`
	echo "${n},${time}" >> $f
    done
}

distributed()
{
    f="timings_distributed.csv"
    echo "n,time" > $f

    for n in `seq 4`; do
	echo "Running with $n core(s)..."
	time=`mpirun --oversubscribe -n $(($n+1)) ./run_distributed.py | grep "Elapsed Time:" | grep -Eo '[0-9]+([.][0-9]+)?'`
	echo "${n},${time}" >> $f
    done
}

distributed
