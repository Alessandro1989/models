from PIL import Image
import os
import sys
from pathlib import Path, PureWindowsPath

def main():
    readInfoAndCropDigits()
    #read_input_train()

def readInfoAndCropDigits():
    data_dir = '/tmp/svhn_data'
    data_dirDigits = '/tmp/svhn_dataDigits'
    pathDataDir = Path(data_dir, 'train')
    listFiles = list(pathDataDir.glob('*.png'))
    size = (128, 128)


    digitsInfo = read_input_train()
    # qui ci occupiamo di tagliare le cifre e fare il dataset

    dir = Path(data_dirDigits)
    if(not dir.exists()):
        dir.mkdir()
    else:
        print("dir digits cropped already exits..")
        return
        #print("cleaning files..")
        #for file in list(dir.glob('*')):
        #    file.unlink()

    iteration = 0
    numberOfImages = len(listFiles)
    apercentage = int(numberOfImages/100)
    print("cropping images:")
    for file in listFiles:
        #file, ext = os.path.splitext(infile)
        fileNamePath = str(file.absolute())
        fileNameImg = fileNamePath.split("\\")[-1]
        infoForDigitsImage = digitsInfo[fileNameImg]

        for singleInfoDigit in infoForDigitsImage:
            im = Image.open(fileNamePath)
            top = singleInfoDigit['top']
            left = singleInfoDigit['left']
            height = singleInfoDigit['height']
            width = singleInfoDigit['width']
            label = singleInfoDigit['label']
            #box – The crop rectangle, as a(left, upper, right, lower) - tuple.
            im = im.crop( (int(left),int(top), int(left)+int(width), int(top)+int(height)) )
            #im = im.resize(size) #thumbnail is better.. -> doesn't work!!!
            #im.thumbnail(size) (never mind, it will do later)..
            fileNameImgForSave = fileNameImg.split(".")[0]+"_" + label + ".png"
            fileToSave = Path(data_dirDigits, fileNameImgForSave)
            im.save(fileToSave, "JPEG")

        iteration +=1
        if(iteration % apercentage == 0):
            print(str(int((iteration/numberOfImages)*100)) + "%", end='...', flush=True)

    if(iteration>=numberOfImages):
        print("100% done")


def read_input_train():
    with open("digitStruct_train.txt", "r") as f:
        digitDict = {}

        for line in f:
            tokens = line.split(';')
            digitsInfo = {}
            for token in tokens:
                if ':' in token:
                    key = token.split(':')[0].strip(' ')
                    value = token.split(':')[1].strip(' ')
                    if key == "name":
                        if value in digitDict:
                            digitDict[value].append(digitsInfo)
                            #listDigits.append(digitsInfo)
                            #digitDict[value] = listdigits
                        else:
                            digitDict[value] = [digitsInfo] #'1.png': {'top':..., 'left':..., 'height':...,...}
                    else:
                        digitsInfo[key] = value

        return digitDict


if __name__ == '__main__':
  main()
