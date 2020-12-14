from downloader import *

import os

dl = Downloader()



def main():
    while not dl.isDone():
        img = dl.downloadNextImage()
        print(img)

    print("Finished downloading and organizing template images!")


main()