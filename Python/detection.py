import numpy as np
import imutils
from imutils.contours import sort_contours
import cv2

def detect(model, path):
	image = cv2.imread(path)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)

	edged = cv2.Canny(blurred, 30, 150)
	cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sort_contours(cnts, method="left-to-right")[0]

	chars = []

	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)

		if (w >= 5 and w <= 350) and (h >= 15 and h <= 320):
			roi = gray[y:y+h, x:x+w]
			thresh = cv2.threshold(roi, 0 ,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			(tH, tW) = thresh.shape

			if tW > tH:
				thresh = imutils.resize(thresh, width = 32)

			else:
				thresh = imutils.resize(thresh, height = 32)

			(tH, tW) = thresh.shape
			dX = int(max(0, 32 - tW) / 2.0)
			dY = int(max(0, 32 - tH) / 2.0)
			# pad the image and force 32x32 dimensions
			padded = cv2.copyMakeBorder(thresh, top=dY, bottom=dY,
				left=dX, right=dX, borderType=cv2.BORDER_CONSTANT,
				value=(0, 0, 0))
			padded = cv2.resize(padded, (32, 32))
			# prepare the padded image for classification via our
			# handwriting OCR model
			padded = padded.astype("float32") / 255.0
			padded = np.expand_dims(padded, axis=-1)
			# update our list of characters that will be OCR'd
			chars.append((padded, (x, y, w, h)))

	# extract the bounding box locations and padded characters
	boxes = [b[1] for b in chars]
	chars = np.array([c[0] for c in chars], dtype="float32")
	# OCR the characters using our handwriting recognition model
	preds = model.predict(chars)

	for (pred, (x, y, w, h)) in zip(preds, boxes):
		# draw the prediction on the image
		print("[INFO] {} - {:.2f}%".format(pred[0], pred[1] * 100))
		cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.putText(image, "{} - {}".format(pred[0], int(pred[1]*100)), (x - 10, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
	# show the image
	cv2.imshow("Image", image)
	cv2.waitKey(0)