#!/bin/bash

for FILE in ~/input/*
do
   cp $FILE data.csv
   right=${FILE#*_}
   dianjia=${right%.*} 
   ./lstm.py $dianjia
   echo $FILE $right $dianjia >> test.out
done
sort -gk 1 -t, predict.csv > result.csv
rm predict.csv
