import datetime

from django.shortcuts import render
from django.http import JsonResponse
# Create your views here.
from django.http import HttpResponse
from django.template import RequestContext, loader
from django.http.response import StreamingHttpResponse
# from pathlib import Path
# from imutils.video import VideoStream
# from imutils.video import FPS
import numpy as np
# import argparse
# import imutils
import time
import cv2
from django.views.decorators.csrf import csrf_exempt
from geopy.geocoders import Nominatim
import simplejson as json


def smartwindow(request):
    template = loader.get_template('main.html')
    return HttpResponse(template.render({}, request))


def video_feed_1(request):
    return StreamingHttpResponse(stream_1(), content_type='multipart/x-mixed-replace; boundary=frame')


def stream_1():
    classes = []

    with open('smartWindow/yolo/coco.names', 'r') as f:
        classes = f.read().splitlines()

    print(classes)

    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--input", required=True,	help="path to input video")
    # ap.add_argument("-o", "--output", required=True,	help="path to output video")
    # ap.add_argument("-y", "--yolo", required=True, help="base path to YOLO directory")
    # ap.add_argument("-c", "--confidence", type=float, default=0.5, help="minimum probability to filter weak detections")
    # ap.add_argument("-t", "--threshold", type=float, default=0.3, help="threshold when applyong non-maxima suppression")
    # args = vars(ap.parse_args())
    #
    #
    print("till here")

    # load the COCO class labels our YOLO model was trained on
    labelsPath = 'smartWindow/yolo/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    # np.random.seed(42)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # derive the paths to the YOLO weights and model configuration
    weightsPath = 'smartWindow/yolo/yolov3-tiny.weights'
    configPath = 'smartWindow/yolo/yolov3-tiny.cfg'

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # loading image
    cap = cv2.VideoCapture(0)  # 0 for 1st webcam
    font = cv2.FONT_HERSHEY_PLAIN
    starting_time = time.time()
    frame_id = 0

    while True:
        _, frame = cap.read()  #
        frame_id += 1

        height, width, channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

        net.setInput(blob)
        outs = net.forward(outputlayers)
        # print(outs[1])

        # Showing info on screen/ get confidence score of algorithm in detecting an object in blob
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.3:
                    # onject detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # cv2.circle(img,(center_x,center_y),10,(0,255,0),2)
                    # rectangle co-ordinaters
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

                    boxes.append([x, y, w, h])  # put all rectangle areas
                    confidences.append(
                        float(confidence))  # how confidence was that object detected and show that percentage
                    class_ids.append(class_id)  # name of the object tha was detected

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 1, (0, 0, 0), 2)

        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        cv2.putText(frame, "FPS:" + str(round(fps, 2)), (10, 50), font, 2, (0, 0, 0), 1)

        # cv2.imshow("Image", frame)
        key = cv2.waitKey(1)  # wait 1ms the loop will start again and we will process the next frame

        if key == 27:  # esc key stops the process
            break;

        cv2.imwrite('demo.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('demo.jpg', 'rb').read() + b'\r\n')


@csrf_exempt
def location(request):
    print("IN geolocationnn")
    # xp = request.GET.get("earned_xp")
    #
    # request.POST.get('data')

    if request.is_ajax and request.method == 'POST':
        # access you data by playing around with the request.POST object
        lat = request.POST['lat']
        long = request.POST['long']
        gps = [str(lat) + "," + str(long)]
        print(gps)
        geolocator = Nominatim(user_agent="SmartRail")
        address = str(geolocator.reverse(gps))
        return JsonResponse({'address': address, 'gps': gps})  # Sending an success response

# def location(request):
#     template = loader.get_template('main.html')
#     return HttpResponse(template.render({}, request))
