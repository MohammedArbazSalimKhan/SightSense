from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import base64
import os
import numpy as np
import time
import cv2
import random

route = os.path.dirname(os.path.abspath(__file__))
static_path = os.path.join(route, "static")
des = os.path.join(static_path, "test.jpeg")
result_path = os.path.join(static_path, "result.jpeg")
#result1_path = os.path.join(static_path, "result1.jpeg")

app = Flask(__name__)


@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')


@app.route('/cam')
def Camera():
    return render_template('cam.html')


@app.route('/saveimage', methods=['POST'])
def saveImage():
    data_url = request.values['imageBase64']
    image_encoded = data_url.split(',')[1]
    body = base64.b64decode(image_encoded.encode('utf-8'))
    file = open(des, "wb")
    file.write(body)
    return "ok"

@app.route('/showimage')
def showImage():
    print("in showImage")
    return render_template('output.html')


@app.route('/process')
def process():
    return render_template('process.html')





@app.route('/output')
def output():
    # load the COCO class labels our YOLO model was trained on
    labelsPath = "coco.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    # initialize a list of colors to represent each possible class label
    np.random.seed(11)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                               dtype="uint8")
    #derive the paths to the YOLO weights and model configuration
    weightsPath = "yolov3.weights"
    configPath = "yolov3.cfg"
    # load our YOLO object detector trained on COCO dataset (80 classes)
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    # load our input image and grab its spatial dimensions
    image = cv2.imread(des)
    (H, W) = image.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in net.getUnconnectedOutLayers()]

    # construct a blob from the input image and then perform a forward
    # pass of the YOLO object detector, giving us our bounding boxes and
    # associated probabilities
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
    # show timing information on YOLO
    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > 0.5:
                # scalethe bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                # use the center (x, y)-coordinates to derive the top and
                #  left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    # apply non-maxima suppression to suppress weak, overlapping bounding
    # boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,
                            0.3)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    dam = 0
    if dam == 1:
        cv2.imwrite(result_path, image)
        imgval = result_path
    else:
        cv2.imwrite(result1_path, image)
        imgval = result1_path

    a = random.randrange(1, 5000, 1)
    b = str(a) + ".jpeg"

    dest = os.path.join(static_path, b)
    cv2.imwrite(dest, image)
    imgval = dest

    return render_template('output.html', imgval=imgval)


if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0', port=2024)