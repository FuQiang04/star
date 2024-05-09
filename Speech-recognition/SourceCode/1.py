import os
from pydub import AudioSegment
import random
import librosa

path = "../DataSource/source-train"

a = -1

for i in range(2,10):
    k = str(i) + '.txt'
    if not os.path.exists(k):
        num = i
        f1 = open(k, 'w')
        f1.close()
        break
'''
for root, dirs, files in os.walk(path):
    if a == -1:
        a = 0
        continue
    for name in files:
        path1 = root + '/' + name
        os.remove(path1)
        break
    if a != 15:
        print("error")
    a = 0

'''
'''
        audio = AudioSegment.from_file(path_mp3, format='mp3')
        if len(audio) > 4000:
            name = root[-5:] + "F{:0>4d}".format(a) + ".wav"
            path_wav = root + '/' + name
            audio.export(path_wav, format="wav")
            a += 1
            if a == 15:
                a = 0
                break
    
    
        path_mp3 = path + '/' + name
        
        
        else:
            path1 = "../DataSource/source-train/" + name.split("#")[1]
            for root1, dirs1, files1 in os.walk(path1):
                for name1 in files1:
                    path1_mp3 = path1 + '/' + name1
                    audio1 = AudioSegment.from_file(path1_mp3, format='mp3')
                    if len(audio1) > 4000:
                        name2 = name.split("#")[1] + ".wav"
                        path_wav = path + '/' + name2
                        audio1.export(path_wav, format="wav")
                        break
                else:
                    print(name.split("#")[1])
'''

