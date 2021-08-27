# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 17:37:56 2021

@author: Sherwood
"""

import cv2
import numpy as np
import utils

path = "1.jpg"
widthImg = 500
heightImg = 500
questions = 5
choices = 5
ans = np.array([1, 2, 0, 1, 4])


img = cv2.imread(path)

#Pre-processing
img = cv2.resize(img, (widthImg, heightImg))
imgCountours = img.copy()
imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray, (5,5), 1)
imgCanny = cv2.Canny(imgBlur, 10, 50)
imgB = img.copy()


#FINDING ALL CONTOURS
contours, hierarchy = cv2.findContours(imgCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cv2.drawContours(imgCountours, contours, -1, (0,255,0), 10)
#contours is list of every contour, each contour include continous points

rectCon = utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])
gradePoints = utils.getCornerPoints(rectCon[1])
# print(gradePoints)
 

if biggestContour.size != 0 and gradePoints.size != 0:
    cv2.drawContours(imgB, biggestContour, -1, (0,255,0), 10)    
    cv2.drawContours(imgB, gradePoints, -1, (0,0,255), 10)  
    
    biggestContour = utils.reoder(biggestContour)
    gradePoints = utils.reoder(gradePoints)
    
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0, 0], [widthImg, 0], [0, heightImg], [widthImg, heightImg]])
    matrix = cv2.getPerspectiveTransform(pt1, pt2)
    imgWarpColored = cv2.warpPerspective(img, matrix, (widthImg, heightImg))
    
    pt1G = np.float32(gradePoints)
    pt2G = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
    matrixG = cv2.getPerspectiveTransform(pt1G, pt2G)
    imgWarpColoredG = cv2.warpPerspective(img, matrixG, (325, 150))
    # cv2.imshow("Grade Box", imgWarpColoredG)
    
    #APPLY THRESHOLD
    imgWarpColored_Gray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWarpColored_Gray, 170, 255, cv2.THRESH_BINARY_INV)[1]
    
    boxes = utils.splitBoxes(imgThresh)
    
    #GETTING NONZERO PIXEL VALUES OF EACH BOX
    myPixelVal = np.zeros((questions,choices))
    i = j = 0 #index of myPixelVal matrix
    
    for bubble in boxes:
        totalPixels = cv2.countNonZero(bubble)
        myPixelVal[i][j] = totalPixels
        j+=1
        if j==choices: i+=1 ; j = 0
            
    print(myPixelVal)
    
    
    #FINDING INDEX VALUES OF THE MARKINGS
    myIndex = []
    for i in range(0, questions):
        idMaxPixel = np.argmax(myPixelVal[i])
        myIndex.append(idMaxPixel)
    
    myIndex = np.array(myIndex)
    
    
    #GRADING
    grade = 1*(ans==myIndex)
    score = grade.mean() * 100
    
    
    #DISPLAY ANSWERS
    imgResult = imgWarpColored.copy()
    imgResult = utils.showAnswers(imgResult, myIndex, grade, ans, questions, choices)
    
    imRawDrawing = np.zeros_like(imgWarpColored)
    imRawDrawing = utils.showAnswers(imRawDrawing, myIndex,grade, ans, questions, choices)
    
    invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
    imgInvWar = cv2.warpPerspective(imRawDrawing, invMatrix, (widthImg, heightImg))
    imgFinal = img.copy()
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWar, 1, 0)
    
    imgRawGrade = np.zeros_like(imgWarpColoredG)
    cv2.putText(imgRawGrade, str(int(score))+"%", (50, 100), cv2.FONT_HERSHEY_SCRIPT_COMPLEX, 3, (0, 255, 255), 3)
       
    invMatrixG = cv2.getPerspectiveTransform(pt2G, pt1G)
    imgInveGrade = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthImg, widthImg))
    imgFinal = cv2.addWeighted(imgFinal, 1, imgInveGrade, 1, 0)

    
    # cv2.imshow("Grade", imgRawGrade)    
    cv2.imshow("Final Img", imgFinal)
    
    # cv2.circle(imgC, (234, 110), 10, (255, 0, 0), cv2.FILLED)
    # cv2.circle(imgC, (174, 369), 10, (255, 87, 45), cv2.FILLED)
    # cv2.circle(imgC, (601, 405), 10, (0, 0, 255), cv2.FILLED)
    # cv2.circle(imgC, (591, 147), 10, (100, 45, 255), cv2.FILLED)
    
    # cv2.circle(imgC, (354, 426), 10, (255, 0, 0), cv2.FILLED)
    # cv2.circle(imgC, (345, 511), 10, (255, 87, 45), cv2.FILLED)
    # cv2.circle(imgC, (604, 530), 10, (0, 0, 255), cv2.FILLED)
    # cv2.circle(imgC, (599, 446), 10, (100, 45, 255), cv2.FILLED)


    
imgBlank = np.zeros_like(img)
imageArray = ([img, imgGray, imgBlur, imgCanny], [imgCountours, imgB, imgWarpColored, imgThresh],
              [imgResult, imRawDrawing, imgInvWar, imgFinal])
imgStacked = utils.stackImages(imageArray, 0.45)

cv2.imshow("Image", imgStacked)
cv2.waitKey(0)
