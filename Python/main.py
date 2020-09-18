import os
import numpy as np
from PIL import Image
from imutils import build_montages
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, MaxPooling2D, Lambda, BatchNormalization, add, multiply
from tensorflow.keras.layers import Conv2DTranspose, GlobalAveragePooling2D,  Dense, Average, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.nn import relu, relu6, swish
from tensorflow.keras.optimizers import SGD
import cv2
# from tensorflow.keras.activations import relu, swish
from model import MinimNet
from detection import detect

class OCR:
	def __init__(self,  data_dir, img_size=(32,32), epochs=20, batch_size=128, load=False, load_path='', labelMode=0):
		'''
			labelMode = 0 is default that consist of symbols, digits, upper and lowercase letters
			labelMode = 1 is only consist of digits uppercase and lowercase
			labelMode = 2 is consist of digits and uppercase
		'''
		self.img_size = img_size
		self.data_dir = data_dir
		self.epochs = epochs
		self.batch_size = batch_size
		self.__init_lr__ = 1e-1

		if labelMode == 0:
			self.labelNames = 	'!"#$%&'+"'()*+,-./"+\
								"0123456789"+\
								":;<=>?"+\
								"ABCDEFGHIJKLMNOPQRSTUVWXYZ"+\
								"abcdefghijklmnopqrstuvwxyz"
		elif labelMode == 1:
			self.labelNames = 	"0123456789"+\
								"ABCDEFGHIJKLMNOPQRSTUVWXYZ"+\
								"abcdefghijklmnopqrstuvwxyz"
		elif labelMode == 2:
			self.labelNames = 	"0123456789"+\
								"ABCDEFGHIJKLMNOPQRSTUVWXYZ"

		self.labelNames = [l for l in self.labelNames]

		self.__load_dataset__()

		self.__create_model__()
		if load:
			self.load_path=load_path
			self.__load_model__()


	def __load_dataset__(self):
		img = []
		lbl = []

		for folder in os.listdir(self.data_dir):
			if os.path.isdir(self.data_dir+'/'+folder):
				for file in os.listdir(self.data_dir+'/'+folder):
					img.append(np.expand_dims(np.array(Image.open(self.data_dir+'/'+folder+'/'+file).resize(self.img_size))*1./255.,axis=-1))
					lbl.append(self.labelNames.index(chr(int(folder))))

		img = np.array(img)
		lbl = np.array(lbl)

		le = LabelBinarizer()
		lbl = le.fit_transform(lbl)

		classTotals = lbl.sum(axis=0)
		self.classWeight = {}

		for i in range(len(classTotals)):
			self.classWeight[i] = classTotals.max() / classTotals[i]

		(self.trainX, self.testX, self.trainY, self.testY) = train_test_split(img,
							lbl, test_size=0.25, stratify=lbl, random_state=42)

		self.aug = ImageDataGenerator(rotation_range=10, zoom_range=.05, 
								width_shift_range=.1, height_shift_range=.1,
								shear_range=.15, horizontal_flip=False,
								fill_mode="nearest")

	def __create_model__(self):
		self.opt = SGD(lr=self.__init_lr__, decay=self.__init_lr__/self.epochs)

		self.model = MinimNet((self.img_size[0], self.img_size[1], 1), len(self.classWeight))

		self.model.compile(loss="categorical_crossentropy", optimizer=self.opt, metrics=["accuracy"])

	def __load_model__(self):
		self.model.load_weights(self.load_path)

	def train_model(self):
		earlyStopping = EarlyStopping(monitor='val_loss', patience=20, verbose=0, mode='min')
		mcp_save = ModelCheckpoint('OCR-2U-{epoch:03d}.h5', save_best_only=True, monitor='val_loss', mode='min')
		reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=1, min_lr=.000001, mode='min')

		self.H = self.model.fit(
			self.aug.flow(self.trainX, self.trainY, batch_size=self.batch_size),
			validation_data=(self.testX, self.testY),
			steps_per_epoch=len(self.trainX)//self.batch_size,
			epochs=self.epochs, callbacks=[earlyStopping, mcp_save, reduce_lr_loss],
			class_weight=self.classWeight, verbose=2)

		self.plot()

	def test_model(self):
		predictions = self.model.predict(self.testX, batch_size=self.batch_size)
		print(classification_report(self.testY.argmax(axis=1),
			predictions.argmax(axis=1), target_names=self.labelNames))

	def predict(self, images):
		predictions = self.model.predict(images)
		preds = []
		for pred in predictions:
			i = np.argmax(pred)
			prob = pred[i]
			preds.append([self.labelNames[i], prob])
		return preds

	def plot(self):
		N = np.arange(0, len(self.H.history["loss"]))
		plt.style.use("ggplot")
		plt.figure()
		plt.plot(N, self.H.history["loss"], label="train_loss")
		plt.plot(N, self.H.history["val_loss"], label="val_loss")
		plt.title("Training Loss and Accuracy")
		plt.xlabel("Epoch #")
		plt.ylabel("Loss/Accuracy")
		plt.legend(loc="lower left")
		plt.savefig("Result-history.png")

	def openCV_test(self):
		images = []
		for i in np.random.choice(np.arange(0, len(self.testY)), size=(49,)):
			
			probs = self.model.predict(self.testX[np.newaxis, i])
			prediction = probs.argmax(axis=1)
			label = self.labelNames[prediction[0]]
			
			image = (self.testX[i] * 255).astype("uint8")
			color = (0, 255, 0)
			
			if prediction[0] != np.argmax(self.testY[i]):
				color = (0, 0, 255)
			
			image = cv2.merge([image] * 3)
			image = cv2.resize(image, (96, 96), interpolation=cv2.INTER_LINEAR)
			cv2.putText(image, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
				color, 2)
			
			images.append(image)
		
		montage = build_montages(images, (96, 96), (7, 7))[0]
		
		cv2.imshow("OCR Results", montage)
		cv2.waitKey(0)


if __name__ == '__main__':
	data = 'curated/curated'
	# model = OCR(data, epochs=100, labelMode=2)
	model_path='OCR-2U-024.h5'
	model = OCR(data, epochs=50, load=True, load_path=model_path, labelMode=2)
	model.train_model()
	model.test_model()
	model.openCV_test()

	image_path = "test_image.jpg"
	detect(model, image_path)