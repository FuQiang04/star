import librosa
from pydub import AudioSegment
import numpy as np
import joblib
import sklearn
import os


if __name__ == "__main__":
    path_mp3 = input()
#    audio = AudioSegment.from_file(path_mp3, format='mp3')
    name = path_mp3[:-4]
    path_wav = path_mp3[:-4] + ".wav"
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
    np.save(file=name+".npy", arr=fea_mfcc)

    data = np.load(name+".npy").T
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