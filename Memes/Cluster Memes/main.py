from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np 
import os
from sklearn.cluster import KMeans,DBSCAN, OPTICS
import cv2
from PIL import Image
import shutil
from distutils.dir_util import copy_tree
import pandas as pd
from sklearn.decomposition import PCA

#Initialize VGG19 model for feature extraction
def init_model(config = {}):
	base_model = VGG19(weights = 'imagenet')
	return Model(inputs = base_model.input, outputs = base_model.get_layer('fc2').output)

#Extract Features using VGG19
def extract_features(img_path = ""):
	for i in os.listdir(img_path):
		if ".DS" in i:
			continue
		img = image.load_img(img_path+i,target_size=(224,224),color_mode = 'rgb')
		x = image.img_to_array(img)
		x = np.expand_dims(x,axis=0)
		x = preprocess_input(x)
		features.append(model.predict(x)[0])
		images.append(i)

#Get cosine similarity between feature vectors of images
def get_cosine_sim(a,b):
	return cosine_similarity(np.array(a).reshape(1,len(a)),np.array(b).reshape(1,len(b)))

#Cluster two images based on cosine similarity
def cluster(img_path,dump_path,config = {}):

	clustering_alg = OPTICS(min_samples = 3, metric = "cosine").fit(features)
	#print(clustering_alg.labels_)

	if not os.path.isdir(dump_path):
		os.mkdir(dump_path)
	
	if os.path.isdir(dump_path):
		for f in os.listdir(dump_path):
			shutil.rmtree(os.path.join(dump_path,f))
	
	for label in clustering_alg.labels_:
		#print(os.path.isdir(dump_path+str(label)))
		if not os.path.isdir(dump_path+str(label)):
			os.mkdir(dump_path+str(label))
			#print(label)

	for i in range(0, len(images)):
		shutil.copy(img_path+images[i],dump_path+str(clustering_alg.labels_[i]))


#Merge unclustered images into a cluster if they are similart o the images in the cluster
def converge_unclustered():
	src_to_dest = dict()
	for unclustered_image in os.listdir(dump_path+"-1/"):
		max = 0
		for cluster in os.listdir(dump_path):
			flag = 0
			if "-1" in cluster:
				continue
			for clustered_image in os.listdir(os.path.join(dump_path,cluster)):
				csm = get_cosine_sim(master_features[clustered_image],master_features[unclustered_image])[0]

				if csm >= 0.90 and csm > max:
					max = csm
					src_to_dest[dump_path+"-1/"+unclustered_image] = os.path.join(dump_path,cluster)
					print(clustered_image)
					
	return src_to_dest


def move_files(src_to_dest):
	for key in src_to_dest:
		shutil.move(key,src_to_dest[key])


#Convert images to grayscale -- not used
def convert_gray():
	for i in os.listdir(img_grayscale):
		try:
			img = Image.open(img_grayscale+i).convert("L")
			img.save(img_path+i)
		except:
			print(i+" Failed")
			continue

#Merge two clusters if they are similar
def merge_clusters():
	iter = os.listdir(dump_path)
	for cluster in iter:
		if cluster == "-1":
			continue
		for c in os.listdir(dump_path):
			if c in cluster or c == "-1":
				continue
			ctr = 0
			sum = 0
			try:
				for image in os.listdir(os.path.join(dump_path,c)):
					for image2 in os.listdir(os.path.join(dump_path,cluster)):
						sum += get_cosine_sim(master_features[image],master_features[image2])[0]
						ctr += 1
			except:
				continue

			if (sum/ctr) >= 0.90:
				print(cluster,str((sum/ctr)))
				copy_tree(os.path.join(dump_path,c),os.path.join(dump_path,cluster))
				shutil.rmtree(os.path.join(dump_path,c))

#PCA to reduce dimentions -- not used
def reduce():
	global features
	pca  = PCA()
	components = pca.fit_transform(features)
	features = pd.DataFrame(data = components).to_numpy()






img_path =  "path to meme graphs" 
dump_path = "path to folder containing clusters"
features = list()
images = list()
master_features = dict()

model = init_model()
#convert_gray()
extract_features(img_path = img_path)
cluster(img_path=img_path,dump_path=dump_path)
#reduce()
master_features = dict(zip(images,features))
src_to_dest = converge_unclustered()
#print(src_to_dest)
move_files(src_to_dest)
merge_clusters()
