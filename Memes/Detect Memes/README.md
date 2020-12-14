# Detect Memes

We use the dataset downloaded under Download Dataset to train our classifier to detect if a graph is a meme or not

We do this by following the steps below:

A. Extract features of an image:

--- We first initialize a pretrained neural network model for this. 

--- The model we are using for feature extraction is VGG19(https://keras.io/api/applications/vgg/#vgg19-function) trained over the imagenet dataset(http://www.image-net.org/)

--- We then infer every image in our training set and extract the outputs from the penultimate layer of VGG19 to get a vectorized representation of the image in our training set.

B. Build a meme classifier:

--- We then train the vectorized images using VGG19 on a one class SVM classfier(https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html). 

--- We then use the classifier to classify images of graphs as meme or not meme ans we store the images of the memes for further steps.






