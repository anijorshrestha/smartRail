import datetime
import urllib

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
import requests, json
import pytemperature


def smartwindow(request):
    template = loader.get_template('main.html')
    return HttpResponse(template.render({}, request))


def video_feed_1(request):
    return StreamingHttpResponse(stream_1(), content_type='multipart/x-mixed-replace; boundary=frame')


address = ""

def test(request):
    template = loader.get_template('stream.html')
    return HttpResponse(template.render({}, request))


def stream_1():

    classes = ['Fraunhofer Society', 'MERGE Research Centre', 'TUC Orange Building']
    today = datetime.datetime.now()
    api_key = "9d151b0f0a64c86337b0094a98ee7132"
    base_url = "http://api.openweathermap.org/data/2.5/weather?"
    city_name = "Chemnitz"
    complete_url = base_url + "appid=" + api_key + "&q=" + city_name
    response = requests.get(complete_url)
    x = response.json()
    print(x)
    if x["cod"] != "404":
        # store the value of "main"
        # key in variable y
        y = x["main"]
        z = x["weather"]
        coord = x["coord"]
        lat = coord["lon"]
        long = coord["lat"]
        # store the value corresponding
        # to the "temp" key of y
        Kelvin = y["temp"]
        global current_temperature
        global weather_description
        current_temperature = str(pytemperature.k2c(Kelvin))[0:2]
        weather_description = str(z[0]["description"])
        icon = z[0]["icon"]
        global iconurl
        iconurl = "http://openweathermap.org/img/w/" + icon + ".png";
        # resp = requests.get(iconurl, stream=True).raw
        # image = np.asarray(bytearray(resp.read()), dtype="uint8")
        # image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        # img_height, img_width, _ = image.shape
        # print(img_height, img_width)

    print("till here")

    # cv2.imshow('image', image)
    # load the COCO class labels our YOLO model was trained on
    labelsPath = 'smartWindow/yolo/coco.names'
    LABELS = open(labelsPath).read().strip().split("\n")

    # initialize a list of colors to represent each possible class label
    # np.random.seed(42)
    # COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # derive the paths to the YOLO weights and model configuration
    weightsPath = 'smartWindow/yolo/yolov4_training_20000.weights'
    configPath = 'smartWindow/yolo/yolov4_trainingg.cfg'

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # loading image
    cap = cv2.VideoCapture(0)  # 0 for 1st webcam
    font = cv2.FONT_HERSHEY_DUPLEX
    starting_time = time.time()
    frame_id = 0
    iwidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float `width`
    iheight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float `height`
    print(iwidth, iheight)

    while True:
        _, frame = cap.read()  #
        frame_id += 1
        output = frame.copy()
        height, width, channels = frame.shape
        # detecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)  # reduce 416 to 320

        net.setInput(blob)
        outs = net.forward(outputlayers)  # print(outs[1])

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
                # print(x,y,w,h)
                global label
                label=""
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]

                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)

                # cv2.rectangle(frame, (x, y-10), (x + w, y + h), color, 2)
                cv2.rectangle(frame, (x, y - 25), (x+w, y-5), (0, 0, 0), -1)
                cv2.putText(frame, label , (x, y - 15), font, 0.3, (255, 255, 255), 1)
                # if label == 'Fraunhofer Society':
                #     text = 'Name: Fraunhofer Society \nLocation: Reichenhainer Straße 88\nFounded: March 26, 1949\nPhone: +49 371 5397-0\nE-Mail-Address:info@iwu.fraunhofer.de'
                #     y0, dy = y, 12
                #
                #     cv2.rectangle(frame, (x, (y + h) + 10), (x + 230, (y + h) + 100), (0, 0, 0), -1)
                #     for i, txt in enumerate(text.split('\n')):
                #         y = (y0 + i * dy)
                #         cv2.putText(frame, txt, (x + 10, (y + h) + 30), font, 0.3, (255, 2555, 255), 1)
                #
                #     continue
                # elif label == 'MERGE Research Centre':
                #     text = 'Name: MERGE Research Centre \nLocation: Reichenhainer Straße 31/33\nFunded by: German Research Foundation (DFG)\nDaten: +49 371 531-13910\nE-Mail-Address: merge@tu-chemnitz.de'
                #     y0, dy = y, 12
                #     cv2.rectangle(frame, (x, (y + h) + 10), (x + 260, (y + h) + 100), (0, 0, 0), -1)
                #     for i, txt in enumerate(text.split('\n')):
                #         y = (y0 + i * dy)
                #         cv2.putText(frame, txt, (x + 10, (y + h) + 30), font, 0.3, (255, 2555, 255), 1)
                #     continue
                # elif label == 'TUC Orange Building':
                #     text = 'Name: TUC Orange Building \nLocation: Reichenhainer Strasse 90\nOpening time: 7:00 a.m. - 8:45 p.m. (Weekdays)'
                #     y0, dy = y, 12
                #     cv2.rectangle(frame, (x, (y + h) + 10), (x + 260, (y + h) + 100), (0, 0, 0), -1)
                #     for i, txt in enumerate(text.split('\n')):
                #         y = (y0 + i * dy)
                #         cv2.putText(frame, txt, (x + 10, (y + h) + 30), font, 0.3, (255, 2555, 255), 1)
                #     continue
        todays = datetime.datetime.now()
        elapsed_time = time.time() - starting_time
        fps = frame_id / elapsed_time
        # cv2.rectangle(frame, (5, 20), (100, 85), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.5, output, 1 - 0.3, 0, output)
        # output[30:30 + image.shape[0], 30:30 + image.shape[1]] = image
        # cv2.putText(output, today.strftime("%B %d, %Y"), (10, 30), font, 0.3, (255, 255, 255), 1)
        # cv2.putText(output, todays.strftime("%H : %M : %S"), (10, 40), font, 0.3, (255, 255, 255), 1)
        # cv2.putText(frame, str(fps), (10, 35), font, 0.5, (0, 0, 0), 1)

        # cv2.putText(output, str(address), (10, 50), font, 0.3, (255,255, 255), 1)

        # cv2.putText(frame, today.strftime("%H : %M : %S"), (10, 85), font, 0.5, (0, 0, 0), 1)
        # cv2.putText(output, current_temperature[0:2] + " " + 'C', (10, 70), font, 0.3, (255, 255, 255), 1)
        # cv2.putText(output, str(weather_description), (10, 80), font, 0.3, (255, 255, 255), 1)

        # cv2.putText(frame,address)
        # cv2.imshow("Image", frame)
        key = cv2.waitKey()  # wait 1ms the loop will start again and we will process the next frame

        if key == 27:  # esc key stops the process
            break

        # button.draw(frame)
        cv2.imwrite('demo.jpg', output)
        # cv2.setMouseCallback('demo.jpg', button.handle_event)
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
        api_key = "OJg8NXsJIEKXA0nwO5gGBJRIqAdneDfX7u-fShxoIyQ"
        base_url = "https://revgeocode.search.hereapi.com/v1/revgeocode?"
        # city_name = "Chemnitz"
        complete_url = base_url + "at=" + lat + "%2C" + long+"&lang=en-US&apikey="+api_key
        response = requests.get(complete_url)
        location = response.json()
        print(location)
        items=location["items"]
        print(items)
        title=items[0]["address"]
        print(title)
        labels=title["label"]
        print(labels)
        addresslist=list(labels.split(","))
        global address
        address = addresslist[0]




        # global gps
        # gps = [str(lat) + "," + str(long)]
        # print(gps)
        # geolocator = Nominatim(user_agent="SmartRail")

        global label
        global iconurl
        global current_temperature
        global weather_description
        print(label)
        # add = str(geolocator.reverse(gps))
        # addresslist = list(add.split(","))
        # address = addresslist[0]
        return JsonResponse({'address': address, 'label': label,'iconurl':iconurl,'currentTemp':current_temperature,'weatherDesc':weather_description})  # Sending an success response

# address, gps = location(request)
# print(address)
# print(gps)
# def location(request):
#     templates = loader.get_template('main.html')
#     return HttpResponse(templates.render({}, request))
