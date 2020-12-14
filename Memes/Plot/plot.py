import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pandas as pd 
import numpy as np
import datetime
import matplotlib.dates as md
import time
import random
import matplotlib

#most popular clusters for plotting
clusters = [82,100,147,9,113,17,14,7,38,57,127]

#get data about tihe time stamp and retweets of each image
path_to_timestamps = "/home/tanush/Desktop/plotting/new_timestamp.csv"
path_to_retweets = "/home/tanush/Downloads/retweet.csv"
path_to_clusters = "/home/tanush/Desktop/plotting/Clusters/Clusters/"

#Store the heroimage for each cluster and their x(timestamp) and y(retweets) values for plotting
hero_image = list()
x_vals = list()
y_vals = list()


timestamps = pd.read_csv(path_to_timestamps)
retweets = pd.read_csv(path_to_retweets,names=["name","retweets"])

#Get the heroimage for each cluster and their x(timestamp) and y(retweets) values for plotting
for cluster in clusters:
	cluster_timestamps = timestamps.loc[timestamps['clusters']==cluster]
	cluster_timestamps = cluster_timestamps.sort_values('time')
	cluster_images = cluster_timestamps["name"].to_list()
	#cluster_images = [path_to_clusters+str(cluster)+"/"+str(image) for image in cluster_images]
	hero_image.append(path_to_clusters+str(cluster)+"/"+str(cluster_images[0]))
	x_vals.append(cluster_timestamps['time'].to_list())
	r_list = [int(retweets.loc[retweets["name"]==image]["retweets"]) for image in cluster_images]
	y_vals.append(r_list)

final = sorted(list(zip(x_vals,y_vals,hero_image)))
final = list(zip(*final))

#Get ready fo plotting
x_vals = final[0]
y_vals = final[1]
hero_image = final[2]
heat_bins = list()
mult = 5

#
for j in range (0,len(y_vals)):
	start = datetime.datetime.strptime("03-01-2020","%m-%d-%Y")+datetime.timedelta(days=7)
	end = datetime.datetime.strptime("05-30-2020","%m-%d-%Y")
	temp = list()
	i=0
	while start<end:
		count = 0
		while i<len(y_vals[j]) and datetime.datetime.strptime(x_vals[j][i],'%Y-%m-%d %H:%M:%S') < start:
			count+=y_vals[j][i]
			i+=1
		temp.append(count+1)
		start = start+datetime.timedelta(days=7)
	count = 0

	if i<len(y_vals[j]):
		while i<=len(y_vals[j])-1:
			count+=y_vals[j][i]
			i+=1
		temp.append(count+1)
	else:
		temp.append(1)
	heat_bins.append(temp)


#Start and end times of your meme clusters
start = md.date2num(datetime.datetime.strptime("03-01-2020","%m-%d-%Y"))
end = md.date2num(datetime.datetime.strptime("05-30-2020","%m-%d-%Y"))




no_of_plots = 11
for i in range(0,no_of_plots,4):

	#below code is for plotting heatmaps for clusters - 4 at a time! 

	'''
	fig = plt.figure()

	ax = fig.add_subplot(2,2,1)

	ax.imshow(np.array(heat_bins[i]).reshape(1,13), extent=[start,end,0,6],cmap=plt.get_cmap("summer"),norm=matplotlib.colors.LogNorm(vmin=1, vmax=260))

	ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d-%y'))
	ax.tick_params(axis='x', labelrotation=45,color='white', labelcolor='white')
	ax.tick_params(axis='y', color='white', labelcolor='white')
	ax.set_ylabel("Cluster "+str(i+1),fontsize = 15)

	ax = fig.add_subplot(2,2,2)

	ax.imshow(np.array(heat_bins[i+1]).reshape(1,13), extent=[start,end,0,6],cmap=plt.get_cmap("summer"),norm=matplotlib.colors.LogNorm(vmin=1, vmax=260))
	ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d-%y'))
	ax.tick_params(axis='x', labelrotation=45,color='white', labelcolor='white')
	ax.tick_params(axis='y', color='white', labelcolor='white')
	ax.set_ylabel("Cluster "+str(i+2),fontsize = 15)

	ax = fig.add_subplot(2,2,3)

	ax.imshow(np.array(heat_bins[i+2]).reshape(1,13), extent=[start,end,0,6],cmap=plt.get_cmap("summer"),norm=matplotlib.colors.LogNorm(vmin=1, vmax=260))
	ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d-%y'))
	ax.tick_params(axis='x', labelrotation=45,color='white', labelcolor='white')
	ax.tick_params(axis='y', color='white', labelcolor='white')
	ax.set_ylabel("Cluster "+str(i+3),fontsize = 15)
	
	ax = fig.add_subplot(2,2,4)

	ax.imshow(np.array(heat_bins[i+3]).reshape(1,13), extent=[start,end,0,6],cmap=plt.get_cmap("summer"),norm=matplotlib.colors.LogNorm(vmin=1, vmax=260))
	ax.xaxis.set_major_formatter(md.DateFormatter('%m-%d-%y'))
	ax.tick_params(axis='x', labelrotation=45,color='white', labelcolor='white')
	ax.tick_params(axis='y', color='white', labelcolor='white')
	ax.set_ylabel("Cluster "+str(i+4),fontsize = 15)
	
	'''


	'''

	x = [datetime.datetime.strptime(timestamp,'%Y-%m-%d %H:%M:%S') for timestamp in x_vals[i]]
	x = md.date2num(x)

    
	#below code is for plotting retweets for clusters witht he hero image in the background- 4 at a time! 
	
	image = plt.imread(hero_image[1])
	fig.add_subplot(2,2,1)
	plt.title("Cluster"+str(i))
	plt.plot_date(x,y_vals[i],linestyle='solid',marker='o')
	plt.imshow(image,extent=[x[0],x[len(x)-1],0,max(y_vals[i])],aspect='auto',alpha = 0.3)

	image = plt.imread(hero_image[i+1])
	fig.add_subplot(2,2,2)
	plt.title("Cluster"+str(i+1))
	plt.plot_date(x,y_vals[0],linestyle='solid',marker='o')
	plt.imshow(image,extent=[x[0],x[len(x)-1],0,max(y_vals[i+1])],aspect='auto',alpha = 0.3)

	image = plt.imread(hero_image[i+2])
	fig.add_subplot(2,2,3)
	plt.title("Cluster"+str(i+2))
	plt.plot_date(x,y_vals[0],linestyle='solid',marker='o')
	plt.imshow(image,extent=[x[0],x[len(x)-1],0,max(y_vals[i+2])],aspect='auto',alpha = 0.3)

	image = plt.imread(hero_image[i+3])
	fig.add_subplot(2,2,4)
	plt.title("Cluster"+str(i+3))
	plt.plot_date(x,y_vals[0],linestyle='solid',marker='o')
	plt.imshow(image,extent=[x[0],x[len(x)-1],0,max(y_vals[i+3])],aspect='auto',alpha = 0.3)

	'''

	fig.tight_layout()
	plt.show()

