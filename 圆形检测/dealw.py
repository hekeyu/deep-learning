import sys
import os
import numpy as np
import cv2

def paintcircle():
    img = np.zeros((64, 64, 3), np.uint8)
    #cv2.circle(img, (32, 32), 32,(0, 0, 255), -1)
    #cv2.imshow("circle", img)
    #cv2.imwrite("circledetect\\"+"11.jpg", img)
    for i in range(50):
        img = np.zeros((64, 64, 3), np.uint8)
        cv2.line(img, (0,0), (np.random.randint(64), np.random.randint(64)), (255,0,0), 5)
        cv2.imwrite("circledetect\\" + "1" + str(i) + ".jpg", img)
    for i in range(50):
        img = np.zeros((64, 64, 3), np.uint8)
        cv2.circle(img, (32, 32), 32, (np.random.randint(255), np.random.randint(255),np.random.randint(255)), -1)
        cv2.imwrite("circledetect\\" + "0" + str(i) + ".jpg", img)
    


def getcircle():
    picdirs = os.listdir("circledetect")
    data = []
    label = np.zeros([len(picdirs), 2])
    for i in range(len(picdirs)):
        data.append(cv2.imread("circledetect\\" + picdirs[i]))
        label[i][int(picdirs[i][0])] = 1
    return data, label
