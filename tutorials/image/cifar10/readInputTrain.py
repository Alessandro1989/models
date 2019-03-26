import os
import sys

def main():
    #dopo averlo convertito con math lab..
    #if not exist error... dont' have.. convert hthe ..bla bla

    positionsDigits = [] #array of dictionary, example: {name:'2.png',top:'4.',left..,label:...}
    with open("digitStruct_train.txt", "r") as f:
        for line in f:
            tokens = line.split(';')
            digitsInfo = {}
            positionsDigits.append(digitsInfo)
            for token in tokens:
                if ':' in token:
                    key = token.split(':')[0].strip(' ')
                    value = token.split(':')[1].strip(' ')
                    digitsInfo[key] = value
                    #if key == "name":

        #adesso abbiamo i dati... ma forse ci conviene più che una lista fare una struttura tipo:
        #nomeImmagine: lista di dizionario info digits..
        #si perchè noi estraiamo un immagine... e da li dobbiamo prendere le informazioni.. quindi..
        #scorrere l'array sarebbe assurdo -> quindi cambiare struttura:

        #with open("digitStruct_train.txt", "r") as f:
        f.seek(0)
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



if __name__ == '__main__':
  main()
