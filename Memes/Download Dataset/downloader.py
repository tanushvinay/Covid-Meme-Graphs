import string
import csv
import requests
import os


class Downloader:
    __memeDataList = []
    __imageURLList = []
    __urlIterator = 0

    def __init__(self):
        self.__populateMemeData()
        self.__populateImageURLList()
        self.__urlIterator = 0

    def __populateMemeData(self):
        memesContained = []
        with open('memegenerator.csv', 'r', encoding='utf-16') as memedata:
            reader = csv.DictReader(memedata,delimiter='\t')
            for row in reader:
                if row['Base Meme Name'] not in memesContained:
                    memesContained.append(row['Base Meme Name'])
                    self.__memeDataList.append(row)

    def __populateImageURLList(self):
        for row in self.__memeDataList:
            self.__imageURLList.append(row['Archived URL'])

    def isDone(self):
        print('Progress:', self.__urlIterator + 1, '/', len(self.__imageURLList))

        if self.__urlIterator >= len(self.__imageURLList):
            return True
        else:
            return False

    def downloadNextImage(self):
        count = self.__urlIterator

        imageFolderPath = os.path.dirname(os.getcwd()) + '/imagefolder'

        if not os.path.isdir(imageFolderPath):
            os.mkdir(imageFolderPath)

        url = self.__imageURLList[count]

        img_data = requests.get(url).content

        if '/' in self.__memeDataList[count]['Base Meme Name']:
            fixedbasename = self.__memeDataList[count]['Base Meme Name'].replace('/', ' ')
        else:
            fixedbasename = self.__memeDataList[count]['Base Meme Name']
        memefilename = imageFolderPath + '/' + fixedbasename + '.jpg'
        with open(memefilename, 'wb') as handler:
            handler.write(img_data)
        self.__urlIterator += 1

        return imageFolderPath + '/' + fixedbasename + '.jpg'