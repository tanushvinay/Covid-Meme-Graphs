# Plotting the clusters 

We use this code to plot two things:

A. The heatmaps for influence(retweets) on a heatmap where each bin is weekly for every cluster:

--- We first store the total number of retweets for every image in every cluster

--- We then sort the images according to their timestamp in ascending order for every cluster

--- We then get the total number of retweets for a one week window(window size can be customized) for every clsuter.

--- Using the binned rewteets value for very cluster we create a heatmap for every cluster. 


B. The influence(retwets) for every image against the timestamp for every image in a cluster

--- We first store the total number of retweets for every image in every cluster

--- We then sort the images according to their timestamp in ascending order for every cluster

--- We then plot for every cluster the timestamp vs influence where every datapoint is an image in the cluster.  

