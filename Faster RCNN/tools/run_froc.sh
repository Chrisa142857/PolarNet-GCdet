#!/bin/bash
# run froc.py
GT_CSV="/home/stat-caolei/code/TCT_V3/statistic_description/tmp/xiugao_test.csv"
PRED_CSV="/home/stat-caolei/code/TCT_V3/tmp/detection_results/all_loc_xiugao_froc_test.csv"

python froc.py \
    $GT_CSV \
    $PRED_CSV
