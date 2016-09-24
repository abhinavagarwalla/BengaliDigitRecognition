import numpy as np

f = open('../results/run1.txt').readlines()
f = [i.strip().split() for i in f]
print f
