import requests
import numpy as np
import cv2 as cv
from gtts import gTTS
import os
##########################################################


def apiLPR():
    url = "https://api.aiforthai.in.th/lpr-v2"
    payload = {'crop': '1', 'rotate': '1'}
    files = {'image': open('frame.jpg', 'rb')}

    headers = {
        'Apikey': "hPVns5FnJHma0iYoJ9TFvIbtLlxAoSlc",
    }

    response = requests.post(url, files=files, data=payload, headers=headers)

    print(response.json())
    A = response.json()
    payload = A[0]
    global tabain
    tabain = payload["lpr"]
    print(tabain)
    GTTS()


def GTTS():
    tts = gTTS(text='รถทะเบียน'+tabain, lang='th')
    tts.save('Hello.mp3')

    file = "Hello.mp3"
    print("Play mp3")
    os.system("mpg123 "+file)


##########################################################
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
    
# while True:
for i in range(60):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame 
    # cv.imshow('gray', gray)
    cv.imshow('frame', frame)
    cv.imwrite("frame.jpg", frame)
    # print("save img successfully")
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# from Russian_LPR import *

try:
    print("API")
    apiLPR()
except:
    print("Error and restart")
    # detectTabain()
    apiLPR()