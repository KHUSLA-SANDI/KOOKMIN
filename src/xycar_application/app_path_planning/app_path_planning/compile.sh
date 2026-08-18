#!/bin/bash

names=("simulator" "motion_planning" "path_planning" "parking")

#sudo apt install cython

for k in "${names[@]}"; 
do 
   echo $k
   cython3 -3 -a $k.py
   gcc -shared -pthread -fPIC -fwrapv -O2 -Wall -fno-strict-aliasing \
       -I/usr/include/python3.10 -o $k.so $k.c
done

cython3 -3 --embed -o main.c main.py
gcc -Os -I /usr/include/python3.10 main.c -lpython3.10 -o main
