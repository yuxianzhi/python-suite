#!/bin/bash

predict_file=/home/xianzhiyu/tensor/tianchi/tielu/predict.csv
cat ${predict_file} | while read line
do
    dianjia_num=${line%%,*}
    norm_file='/home/xianzhiyu/input/dianjia_'${dianjia_num}'.csv'
    rm ${norm_file}
done
