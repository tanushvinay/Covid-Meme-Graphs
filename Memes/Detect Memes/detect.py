#Import Necessary Libraries
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.svm import OneClassSVM
import numpy as np 
import os
from sklearn.cluster import KMeans,DBSCAN, OPTICS
import cv2
from PIL import Image
import shutil
from distutils.dir_util import copy_tree
import pandas as pd
from sklearn.decomposition import PCA
import pickle


#Initiate the VGG19 model and return the outputs of the last fully connected layer
def init_model(config = {}):
	base_model = VGG19(weights = 'imagenet')
	return Model(inputs = base_model.input, outputs = base_model.get_layer('fc2').output)



#We use the VGG19 model to extract features of the images after rescaling the images to (244,244)
def extract_features(img_path = ""):
	for i in os.listdir(img_path):
		if ".DS" in i:
			continue
		try:
			img = image.load_img(img_path+i,target_size=(224,224),color_mode = 'rgb')
		except:
			continue
		x = image.img_to_array(img)
		x = np.expand_dims(x,axis=0)
		x = preprocess_input(x)
		features.append(model.predict(x)[0])
		images.append(i)


#Move images from source to destination
def move_files(src_to_dest):
	for key in src_to_dest:
		shutil.move(key,src_to_dest[key])


#reduce features of the images -- not used
def reduce():
	global features
	pca  = PCA()
	components = pca.fit_transform(features)
	features = pd.DataFrame(data = components).to_numpy()




#Path to meme data 
img_path =  "/home/tvinay/Sampling/MemeDetector/memes/" 
graph_image_path = "/home/tvinay/Sampling/MemeDetector/graphs/"

#storeimage names and extracted features of the images
features = list()
images = list()
master_features = dict()

#init the VGG19 model
model = init_model()
extract_features(img_path = img_path)
#reduce()
master_features = dict(zip(images,features))

#Init the one class SVM classifier with the features of the images
clf = OneClassSVM().fit(features)
#print(clf.score_samples(features))
#print(clf.predict(features))

#Save the classifier for future uses if need be
pickle.dump(clf, open("weights.sav", 'wb'))


#Use the one class SVM classifier to dectect if the image is a meme or not
for i in os.listdir(test_path):
	try:
		img = image.load_img(test_path+i,target_size=(224,224),color_mode = 'rgb')
	except:
		continue
	x = image.img_to_array(img)
	x = np.expand_dims(x,axis=0)
	x = preprocess_input(x)
	x = model.predict(x)[0]
	prediction = clf.predict(x.reshape(1,-1))
	if prediction[0] == 1:
		shutil.copy(test_path+i,'/home/tvinay/Sampling/MemeDetector/detected/')
