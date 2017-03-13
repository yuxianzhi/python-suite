#!/usr/bin/python
# -*- coding: utf-8 -*

import sys
def array_out(array):
        dianjia=sys.argv[1]

        f = file('predict.csv','aw')
        f.write(dianjia)
        for i in array:
                f.write(",")
		f.write(str(int(round(i))))
	f.write('\n')
        f.close()

b=[2.1,2.2,3.5,4.0]
array_out(b)
