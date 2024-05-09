import pyaudio
import wave
import time
import librosa
import numpy as np
import joblib
from pydub import AudioSegment
import sklearn
import os
import random


if __name__ == "__main__":
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = 6
    name = random.randint(1, 1000000)
    WAVE_OUTPUT_FILENAME = "{}.wav".format(name)
    root_path = "../DataSource/temporary/"
    file = root_path + WAVE_OUTPUT_FILENAME
    while  os.path.exists(file):
        name = random.randint(1, 1000000)
        WAVE_OUTPUT_FILENAME = "{}.wav".format(name)
        root_path = "../DataSource/temporary"
        file = root_path + WAVE_OUTPUT_FILENAME
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    print("程序将在三秒后开始录音")
    totalTime = 3
    while totalTime > 0:
        print('%d' % totalTime)
        time.sleep(1)
        totalTime -= 1
    print("开始录音,请说话......")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("录音结束,您的语音数据已录入!")

    stream.stop_stream()
    stream.close()
    p.terminate()
    # 生成说话人的特征文件npy，
    wf = wave.open(root_path + WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()



    path_wav = file
#    audio.export(path_wav, format="wav")
    y, fs = librosa.load(path_wav, sr=None, mono=True)
    # 进行MFCC特征的提取
    raw_mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=19, n_fft=512, hop_length=160, win_length=320, n_mels=20)
    #    raw_mfccs = librosa.feature.mfcc(y=y, sr=sr, S=None, n_mfcc=3, dct_type=2, norm='ortho')
    # 增加动态特征
    raw_mfcc = librosa.util.normalize(raw_mfcc)
    mfcc_delta = librosa.feature.delta(raw_mfcc)
    mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)

    # 拼接生成最终的MFCC特征
    fea_mfcc = np.concatenate([raw_mfcc, mfcc_delta, mfcc_delta2], axis=0)
    name = str(name)
    np.save(file="fea/temporary/" + name+".npy", arr=fea_mfcc)

    data = np.load("fea/temporary/" + name+".npy").T
    path_model = '../DataSource/models'
    for root, dirs, files in os.walk(path_model):
        score_highest = 0
        for model in files:
            gmm = joblib.load(os.path.join(path_model, model))
            score_gmm = gmm.score(data)
            if score_gmm > score_highest:
                spk = model[:-6]
                score_highest = score_gmm
        print("speaker:{}".format(spk))