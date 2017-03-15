#!/bin/bash

predict_file=./predict.csv
cat ${predict_file} | while read line
do
    dianjia_num=${line%%,*}
    norm_file='~/input/dianjia_'${dianjia_num}'.csv'
    rm ${norm_file}
done
