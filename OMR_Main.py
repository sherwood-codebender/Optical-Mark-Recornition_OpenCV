# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:37:56 2021

@author: Sherwood
"""

import cv2
import numpy as np

path = "1.jpg"
img = cv2.imread(path)

cv2.imshow("Original", img)
cv2.waitKey(0)