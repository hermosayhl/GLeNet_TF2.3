import os
import cv2
import sys
import math
import numpy
import random
import datetime






class Timer:
    def __enter__(self):
        self.start = datetime.datetime.now()

    def __exit__(self, type, value, trace):
        _end = datetime.datetime.now()
        print('耗时  :  {}'.format(_end - self.start))


