#!/bin/bash

# Loop through seed values 0 to 9
for seed in {0..9}
do
    echo "Running with seed $seed ..."
    python  main.py --fname configs/cnd_gtb10_ep300.yaml --seed $seed --devices cuda:0
done

