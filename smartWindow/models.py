from django.db import models

# Create your models here.
from django.db import models
import datetime

from django.contrib import messages
from django.db import models
from django.shortcuts import render, redirect
from django.http import JsonResponse, request
# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse, HttpResponseServerError
# from pathlib import Path
# from imutils.video import VideoStream
# from imutils.video import FPS
import numpy as np
# import argparse
# import imutils
import time
import cv2
from geopy.geocoders import Nominatim
import simplejson as json

# Create your models here.
# from smartWindow.views import retrieveData

with open('smartWindow/yolo/coco.names', 'r') as f:
    classes = f.read().splitlines()
labelsPath = 'smartWindow/yolo/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# derive the paths to the YOLO weights and model configuration
weightsPath = 'smartWindow/yolo/yolov3-tiny.weights'
configPath = 'smartWindow/yolo/yolov3-tiny.cfg'
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
layers = net.getLayerNames()
output = [layers[i[0] - 1] for i in net.getUnconnectedOutLayers()]
font = cv2.FONT_HERSHEY_PLAIN


class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, image = self.video.read()
        h, w, c = image.shape
        blob = cv2.dnn.blobFromImage(image, 0.00392, (320, 320), (0, 0, 0), True, crop=False)

        net.setInput(blob)
        out = net.forward(output)

        class_ids = []
        boxex = []
        confidences = []
        for o in out:
            for e in o:
                points = e[5:]
                class_id = np.argmax(points)
                confidence = points[class_id]
                if confidence > 0.6:
                    center_x = int(e[0] * w)
                    center_y = int(e[1] * h)
                    wid = int(e[2] * w)
                    hei = int(e[3] * h)

                    # Drawing the rectangles
                    x = int(center_x - wid / 2)
                    y = int(center_y - hei / 2)

                    boxex.append([x, y, wid, hei])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        total_objects = len(boxex)

        for i in range(total_objects):
            x, y, wid, hei = boxex[i]
            label = str(classes[class_ids[i]])
            # object1 = Object(objectName=label, detectedTime=str(datetime.datetime.now()))
            # object1.save()
            cv2.rectangle(image, (x, y), (x + wid, y + hei), (255, 0, 0), 2)
            cv2.putText(image, label, (x, y + 20), font, 1, (0, 255, 0), 2)
            cv2.resize(image, (300, 300))
        ret, jpeg = cv2.imencode('.jpg', image)
        # retrieveData(request)
        return jpeg.tobytes()


class Object(models.Model):
    objectName = models.CharField('objectName', max_length=200)
    detectedTime = models.DateTimeField('timestamp')

    def __str__(self):
        return self.objectName
