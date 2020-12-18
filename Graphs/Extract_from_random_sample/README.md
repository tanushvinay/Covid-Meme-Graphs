# Extract Covid Images from Random Sampled Tweets

## random_tweets.py
This script has two main tasks:

A. Filter relevant tweets

--- The intuition of Tweets having Covid graphs will have some text related to graphs was used.

--- This helped to filter out irrelevant tweets and decrease computational load

B. Classfying images

--- The media links of the tweets having images were passed to the trained CNN model.

--- A cut-off of -0.05 was set to determine if the image was graph or not

--- The tweets didn't have a unique ID affiliated to them. So, to be able to retrieve the information of the saved images, they were named using the attirbute 'Vertex1' + 'ImportedID'.

The script also saves the URLS, and some other useful attributes for futher analysis


## retrieve_timestamps.py

This script is used to generate a CSV of the timestamps of all the extracted images. The timestamps were necessary for plotting the graphs.


### Requirements ###
1. Python 3.6.
2. Libraries: Numpy, PyTorch, Tensorflow, Pandas, Pillow

### Installation ###
1. pip install numpy
2. pip install pytorch
3. pip install tensorflow
4. pip install pandas 
5. pip install pillow
