import cv2
from cv2 import cornerSubPix
import numpy as np

def empty():
        pass

img = cv2.imread("c:/Temp/Teste.jpg")

#cv2.imshow("Tomate", img)

blur = cv2.GaussianBlur(img, (15,15), 0)

hsv = cv2.cvtColor( blur, cv2.COLOR_BGR2HSV)

#cv2.imshow("Tomate HSV", hsv)

kernel = np.ones((3,3), np.uint8)

#lower = np.array([8,81,0])
#upper = np.array([9,255,255])

r1 = cv2.inRange(hsv, (1,50,200), (10,255,255))
r2 = cv2.inRange(hsv, (160,50,200), (179, 255,255))

mask = cv2.add(r1,r2) 
#cv2.inRange(hsv, lower, upper)

cv2.imshow("Tomate MASK", mask)

Res = cv2.bitwise_and(img, img, mask=mask)

#cv2.imshow("Tomate Res", Res)

canny = cv2.Canny( Res, 100, 100)

cv2.imshow("Tomate EDGE", canny)

teste = cv2.dilate(canny,kernel,iterations = 4)

#cv2.imshow("Tomate Dilate 4", teste)

teste2 = cv2.erode(teste,kernel,iterations = 4)

#cv2.imshow("Tomate ERODE 4", teste2)

contours, _ = cv2.findContours(teste2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(teste2, contours, 0, (0,0,255), 1)

#cv2.imshow("Tomate Contours", teste2)


for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 3)
    M = cv2.moments(c)

cv2.imshow("Tomates", img)

cv2.waitKey(0) 
