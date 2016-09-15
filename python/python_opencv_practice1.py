#!/usr/bin/env python

import numpy as np
import cv2

img = cv2.imread('/usr/share/icons/gnome/128x128/apps/libreoffice-base.png',
             cv2.IMREAD_GRAYSCALE)
img_large = cv2.resize(img, (240, 240))
cv2.imshow('test window tilte', img_large)
cv2.waitKey(0)
cv2.destroyAllWindows()


