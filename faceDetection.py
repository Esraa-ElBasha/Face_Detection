

'''
Face Detection:
This code capture a picture once a face is detected and saves it with the name
 Indicated in the code and in the file indicated here.
 
creators:
 ESraa El-Basha
 Marwan Ibrahim
 
'''


import numpy as np
import cv2
import time

#import serial

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

cap = cv2.VideoCapture(0)

#ser=serial.Serial("COM10",9600,timeout=0)
l=[0,0,0,0]

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    #eyes=eye_cascade.detectMultiScale(gray,2,5)
    #for (x,y,w,h) in eyes:
     #   cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
      #  cv2.putText(img,'eye',(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(200,200,200),2)  
    
    for (x,y,w,h) in faces:
         l=[x,y,w,h]
         print(l)
         print("old1 x",x)
         print("old1 y" ,y)
         cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
         cv2.putText(img,'face',(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(200,200,200),2)
        
    if any(l):   #face detected
            print("mlyana")
            print("old2 x", x)
            print(" old2 y", y)   
            file = "E:\ESHTA\image.png"         #captured image path
            cv2.imwrite(file, img)
            file1 = "E:\ESHTA\gray_image.png"   #gray image path
            #time.sleep(3)                       # delays for 3 seconds
            cv2.imwrite(file1, gray)
            cropped = gray[y:(y+h),x:(x+w)] #cropping
            cv2.imshow("cropped", cropped)
            file = "E:\ESHTA\cropped.png"   
            cv2.imwrite(file, cropped)
            print("old3 x", x)
            print(" old3 y", y)
            break
        
    else:
           print("no face detected yet")

       #drawing a rectangle around the detected face     
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(img,'face',(x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1,(200,200,200),2)

    #ser.write("x")
    #print ser.readline()
    
   # print("7ozn kbeeer")
    cv2.imshow('img',img)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
