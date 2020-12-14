# Cluster Memes

After classifying graphs as memes, we take the "meme graphs" and cluster them based on similarity

We do this by following the steps below:

A. Extract features of an image:

--- We first initialize a pretrained neural network model for this. 

--- The model we are using for feature extraction is VGG19(https://keras.io/api/applications/vgg/#vgg19-function) trained over the imagenet dataset(http://www.image-net.org/)

--- We then infer every image in our training set and extract the outputs from the penultimate layer of VGG19 to get a vectorized representation of the image in our training set.

B. Do a first pass clustering:

--- We then cluster the vectorized images using Optics clustering algotithm(https://scikit-learn.org/stable/modules/generated/sklearn.cluster.OPTICS.html). 

--- We found that using cosine similarity as the similarity measure and the min_samples as 3 to get the best quality clusters.


C. Merge un-clustered images:

--- We then for every unclustered image take the average cosine similarity of the image with every image in a cluster.

--- If the average cosine similarity of the image and the cluster >0.9 we add it to that cluster. 

D. Merge un-clustered images:

--- We then for every image in a cluster, we take the average cosine similarity of the image with every image in another cluster.

--- If the average cosine similarity between the clusters >0.9 we merge the clusters. 

E. Report the X Largest clusters: 

--- We then printout the largest clusters among the clusters for plotting. 

