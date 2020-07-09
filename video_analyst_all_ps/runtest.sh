#!/usr/bin/env bash


for SR_size in {311..431..8}
do
      echo ${factor} ${SR_size}
      python ./main/test.py --config experiments/siamfcpp/test/vot/siamfcpp_googlenet.yaml --SR_size ${SR_size}
done
