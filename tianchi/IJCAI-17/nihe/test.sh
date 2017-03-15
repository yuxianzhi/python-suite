#!/bin/bash

for FILE in ~/input/*
do
   cp $FILE data.csv
   right=${FILE#*_}
   dianjia=${right%.*} 
   ./nihe.py $dianjia
   echo $FILE $right $dianjia >> test.out
done
