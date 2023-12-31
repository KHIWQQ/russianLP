import cv2
import easyocr
import time
import calendar
import multiprocessing as mp
import requests

cap = cv2.VideoCapture(0)

global arTabain
arTabain = []

def detectCam():
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    else:
        print("Webcam is ready")

def easyOCR(gray_plates):
    reader = easyocr.Reader(['th'])  # this needs to run only once to load the model into memory
    result = reader.readtext(gray_plates)

    if result and len(result) > 0:  # Check if result is not empty
        global detected_text
        detected_text = result[0][1]
        global arTabain
        arTabain.append(detected_text)
        print(arTabain)
        print('ข้อความที่ตรวจจับได้:', detected_text)
        return detected_text  # Return the detected text

def detectTabain():
    tabialArray = []
    global captureStatus
    captureStatus = True
    while True:
        ret, frame = cap.read()
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # cv2.imshow('Input', frame)

        # convert input image to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        newgray = cv2.threshold(gray, 0 ,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # read haarcascade for number plate detection
        cascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')
        # Detect license number plates
        plates = cascade.detectMultiScale(gray, 1.1, 4)
        cv2.imshow('Input', newgray)
        if not type(plates) is tuple :
            print("found tabian")
            timenow = time.time()
            [x,y,w,h] = plates[0]
            # draw bounding rectangle around the license number plate on the frame
            cv2.rectangle(frame, (x, y-30), (x+w+30, y+h), (0, 0, 255), 2)
            cv2.putText(frame, "tabain", (x, y-35),
            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
            gray_plates = newgray[y-30:y+h+70, x:x+w+30]
            color_plates = frame[y:y+h, x:x+w]   
            cv2.imshow('Number Plate Image', frame)
            cv2.imshow("new gray",gray_plates)
            if int(timenow*10) % 15 == 0 :
                tabialArray.append(gray_plates)
                print("SAVE TABIAN",len(tabialArray))
                
                if len(tabialArray) >= 3:
                    print("SAVE DONE <<>>")
                    for item in tabialArray:
                        easyOCR(item)
                    # tabialArray = []

                    
        c = cv2.waitKey(1)
        if c == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

detectTabain()
