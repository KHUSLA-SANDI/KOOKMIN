#!/bin/bash

cython3 -3 --embed -o app_cam_exposure.c app_cam_exposure.py
gcc -Os -I /usr/include/python3.10 app_cam_exposure.c -lpython3.10 -o app_cam_exposure
