from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import os

def webcam_mask_detection(frame, face_detector, mask_detector):

	# take the first two elements (h, w) from img.shape tuple
	(h, w) = frame.shape[:2]

	# set scaling factor, spatial size, and mean subtraction values for the blob
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104, 177, 123))

	# feed the blob to the face detector
	face_detector.setInput(blob)
	detections = face_detector.forward()

	# Create lists for face frames, the face locations, and the prediction data
	faces = []
	locs = []
	preds = []

	# loop over the detections
	for i in range(0, detections.shape[2]):
		# extract the confidence associated with the detection
		confidence = detections[0, 0, i, 2]

		if confidence > 0.5:
            # compute the XY-coordinates of the bounding box
			startX = int(detections[0, 0, i, 3] * w)
			startY = int(detections[0, 0, i, 4] * h)
			endX = int(detections[0, 0, i, 5] * w)
			endY = int(detections[0, 0, i, 6] * h)

			# ensure the bounding boxes fall within the dimensions of
			# the frame
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# extract the face region (ROI), convert it from BGR to RGB channel
			# and follow by data preprocessing
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# add the faces and bounding box locations to their lists
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# feed the face image to the mask detector
	if len(faces) > 0:
		# make batch predictions on faces at the same time
		faces = np.array(faces, dtype="float32")
		preds = mask_detector.predict(faces, batch_size=32)

	# return 2-tuples of the face locations and predictions
	return (locs, preds)

# load the serialized face detector
# https://github.com/simplesaad/FaceDetection_Realtime
prototxtPath = os.path.join("face_detector", "deploy.prototxt")
weightsPath = os.path.join("face_detector", "res10_300x300_ssd_iter_140000.caffemodel")
face_detector = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model
mask_detector = load_model("mask_detector.model")

# initialize the video stream
print("OpenCV video streaming...")

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

# loop over the frames from the video stream
while True:
    # success is a boolean value
	success, frame = cap.read()

	# feed frame data into the detection function
	(locs, preds) = webcam_mask_detection(frame, face_detector, mask_detector)

	# loop over the detected face locations and the mask predictions
	for (box, pred) in zip(locs, preds):
		(startX, startY, endX, endY) = box
		(mask, nomask) = pred

		# set the class label text and colors
		label = "GOOD JOB!" if mask > nomask else "PUT ON THE MASK MATE!"
		color = (0, 255, 0) if label == "GOOD JOB!" else (0, 0, 255)

		# include the probability in the label
		label = "{}: {:.2f}%".format(label, max(mask, nomask) * 100)

		# display the label and bounding box on the output frame
		cv2.putText(frame, label, (startX, startY - 20),
			cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# show the output frame
	cv2.imshow("Frame", frame)

	# press `q` key then break the loop
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

cv2.destroyAllWindows()
cap.stop()