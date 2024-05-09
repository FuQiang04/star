import pyaudio
import wave
import time
import librosa
import os
import numpy as np
import joblib
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 16000
RECORD_SECONDS = 15
name = input()
WAVE_OUTPUT_FILENAME = "{}.wav".format(name)
root_path= "../DataSource/user/"
spk_npy_path="fea/user/"
file=root_path+WAVE_OUTPUT_FILENAME
p = pyaudio.PyAudio()
os.makedirs(spk_npy_path,exist_ok=True)
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)
print("程序将在三秒后开始录音")
totalTime=3
while totalTime>0:
        print('%d'% totalTime)
        time.sleep(1)
        totalTime-=1
print("开始录音,请说话......")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("录音结束,您的语音数据已录入!")

stream.stop_stream()
stream.close()
p.terminate()
#生成说话人的特征文件npy，
wf = wave.open(root_path+WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
y,fs = librosa.load(file,sr=None, mono=True)
raw_mfcc = librosa.feature.mfcc(y=y,
                               sr=fs,
                               n_mfcc=19,
                               n_fft=512,
                               hop_length=160,
                               win_length=320,
                               n_mels=20
                               )
raw_mfcc = librosa.util.normalize(raw_mfcc)
mfcc_delta = librosa.feature.delta(raw_mfcc)
mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)
fea_mfcc = np.concatenate([raw_mfcc,mfcc_delta,mfcc_delta2],axis=0)
file_fea = "fea/user/{}.npy".format(name)
np.save(file=file_fea,arr=fea_mfcc)
#生成说话人的GMM
def GMM_MAP(ubm_model, data):
    xdim = data.shape[1]
    T = data.shape[0]
    M = ubm_model.n_components
    ubm_weights = ubm_model.weights_
    ubm_means = ubm_model.means_
    ubm_covars = ubm_model.covariances_
    posterior_prob = ubm_model.predict_proba(data)
    pr_i_xt = (ubm_weights * posterior_prob) / np.asmatrix(np.sum(ubm_weights \
                                                                  * posterior_prob, axis=1)).T
    n_i = np.asarray(np.sum(pr_i_xt, axis=0)).flatten()  # [M, ]
    E_x = np.asarray([(np.asarray(pr_i_xt[:, i]) * data).sum(axis=0) / n_i[i] for i in range(M)])  # [M x xdim]
    E_x2 = np.asarray([(np.asarray(pr_i_xt[:, i]) * (data ** 2)).sum(axis=0) / n_i[i] for i in range(M)])  # [M x xdim]
    relevance_factor = 16
    scaleparam = 1
    alpha_i = n_i / (n_i + relevance_factor)
    new_weights = (alpha_i * n_i / T + (1.0 - alpha_i) * ubm_weights) * scaleparam
    alpha_i = np.asarray(np.asmatrix(alpha_i).T)
    new_means = (alpha_i * E_x + (1. - alpha_i) * ubm_means)
    new_covars = alpha_i * E_x2 + (1. - alpha_i) * (ubm_covars + (ubm_means ** 2)) - (new_means ** 2)
    ubm_model.means_ = new_means
    ubm_model.weights_ = new_weights
    ubm_model.covariances_ = new_covars
    return ubm_model
model_path = '../DataSource/models'
path_fea = 'fea/user/'
path_model = '../DataSource/models/'
datas=[]
data = np.load(os.path.join(path_fea, name + ".npy"))
datas.append(data)
datas = np.concatenate(datas, axis=1).T
ubm = joblib.load(os.path.join(model_path, 'ubm.model'))
gmm = GMM_MAP(ubm, datas)
joblib.dump(gmm, os.path.join(model_path, name + '.model'))
