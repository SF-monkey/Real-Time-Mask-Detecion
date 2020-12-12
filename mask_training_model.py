from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

# setting the learnging rate, epochs and batch size.
LR = 0.0001
Epochs = 10
BS = 30

# dataset from https://github.com/X-zhangyang/Real-World-Masked-Face-Dataset
dir = os.getcwd()
dataset_path = os.path.join(dir, 'RMFD')
label_type = ["masked", "nomask"]

# create two lists for storing image and label data
data = []
labels = []

for i in label_type:
    path = os.path.join(dataset_path, i)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224))
    	image = img_to_array(image)
    	image = preprocess_input(image)

		# filling the image and label data into the lists
    	data.append(image)
    	labels.append(i)

# convert labels into binary categorical value
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# create numpy arrays
data = np.array(data, dtype="float32")
labels = np.array(labels)

# split the dataset into 80% train / 20% test
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.2, stratify=labels, random_state=1)

# split again, create 60% train / 20% validation
(trainX, valX, trainY, valY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=1)

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.2,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.2,
    horizontal_flip=True,
	fill_mode="nearest")

# load the MobileNetV2 network, leaving off the fully-connected head
baseModel = MobileNetV2(weights="imagenet",
						include_top=False,
						input_tensor=Input(shape=(224, 224, 3)))
                        
# freeze the convolutional base so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
	layer.trainable = False

# construct the head of the model on top of the the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
# opt out the dropout layer here to get better result
# headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)

# build a model by chaining together the base_model and feature extractor layers
model = Model(inputs=baseModel.input, outputs=headModel)

# compile the model
opt = SGD(lr=LR, momentum=0.9, nesterov=True, decay=LR / Epochs)
model.compile(loss="binary_crossentropy",
			  optimizer=opt,
			  metrics=["accuracy"])

# show a summary of the model
print(model.summary())

# train the model
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(valX, valY),
	validation_steps=len(valX) // BS,
	epochs=Epochs)

# make predictions on the testing set
predIdxs = model.predict(testX, batch_size=BS)

# for each image in the testing set, find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)

# generate a classification report
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# save the serialize model to file
model.save("mask_detector.model", save_format="h5")

# plot the training loss and accuracy
N = Epochs
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")