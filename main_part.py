import cv2
import imutils
import numpy as np
from gtts import gTTS
import  os
import playsound
import pygame
import serial
import time

# Initialize pygame mixer
pygame.mixer.init()
port = serial.Serial("COM3", 9600, timeout=0.1) 
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

cap = cv2.VideoCapture(0)

def tts(text, lang):
     file = gTTS(text = text, lang = lang)
     file.save("speak_func.mp3")
     playsound.playsound('speak_func.mp3', True)
     os.remove("speak_func.mp3")
     
while True:
    ret, img = cap.read()
    img = imutils.resize(img, width=400)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing information on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            if label == "elephant":
                print("elephant detected")
                port.write(str.encode("A"))
                time.sleep(2)
                #tts("நபர்",'ta')
            elif label == "horse":
                print("Horse detected")
                port.write(str.encode("A"))
                time.sleep(2)
                #tts("கைப்பேசி",'ta')
            elif label == "bear":
                print("Bear detected")
                port.write(str.encode("A"))
                time.sleep(2)
                #tts("மடிக்கணினி",'ta')
            elif label == "giraffe":
                print("Giraffe detected")
                port.write(str.encode("A"))
                time.sleep(2)
                #tts("நாற்காலி",'ta')
    # Show the output frame
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF

    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        break

# Release pygame mixer
pygame.mixer.quit()
cv2.destroyAllWindows()
