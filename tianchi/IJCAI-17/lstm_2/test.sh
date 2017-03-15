#!/bin/bash

for FILE in ~/input/*
do
   cp $FILE stock_dataset.csv
   right=${FILE#*_}
   dianjia=${right%.*} 
   ./stock_predict.py $dianjia
   echo $FILE $right $dianjia >> test.out
done
