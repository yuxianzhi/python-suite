#!/bin/bash

predict1_file=/home/xianzhiyu/tensor/tianchi/tielu/predict.csv
predict2_file=/home/xianzhiyu/tensor/tianchi/nihe/predict.csv
temp_file=./temp.csv


line_num=$1
line_num_format=${line_num}'p'
ori=$(sed -n ${line_num_format} ${predict1_file})
echo $ori
line=${ori#*,}
result=${line//,/\\n};
echo "predict" > ${temp_file}
echo -e $result >> ${temp_file}
scp ${temp_file} yuxianzhi@211.87.224.242:~/python/matplotlib/predict1.csv

line_num=$1
line_num_format=${line_num}'p'
ori=$(sed -n ${line_num_format} ${predict2_file})
echo $ori
line=${ori#*,}
result=${line//,/\\n};
echo "predict" > ${temp_file}
echo -e $result >> ${temp_file}
scp ${temp_file} yuxianzhi@211.87.224.242:~/python/matplotlib/predict2.csv
rm ${temp_file}

dianjia_num=${ori%%,*}
norm_file='/home/xianzhiyu/input/dianjia_'${dianjia_num}'.csv'
echo ${norm_file}
scp ${norm_file} yuxianzhi@211.87.224.242:~/python/matplotlib/normalized_data.csv
