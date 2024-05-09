import librosa
import os
import numpy as np

# 提取 UBM特征

fea_train_path = "fea/train"
os.makedirs(fea_train_path,exist_ok=True)

file_lines = np.loadtxt('train.scp',dtype='str',delimiter=" ")
files= file_lines[:,0]
spk_ids = file_lines[:,1]
utt_ids = file_lines[:,2]

f = open("log.txt", 'a')
time = 0

for file,spk,utt in zip(files,spk_ids,utt_ids):
    # 读取音频文件
    y,fs = librosa.load(file,sr=None, mono=True)
    time += librosa.get_duration(path=file)
    # 进行MFCC特征的提取
    raw_mfcc = librosa.feature.mfcc(y=y,
                                   sr=fs,
                                   n_mfcc=19,
                                   n_fft=512,
                                   hop_length=160, 
                                   win_length=320,
                                   n_mels=20
                                   )
    # 增加动态特征
    raw_mfcc = librosa.util.normalize(raw_mfcc)
    mfcc_delta = librosa.feature.delta(raw_mfcc)
    mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)

    # 拼接生成最终的MFCC特征
    fea_mfcc = np.concatenate([raw_mfcc,mfcc_delta,mfcc_delta2],axis=0)

    # fea_mean = np.mean(fea_mfcc,axis=1,keepdims=True)
    # fea_std = np.std(fea_mfcc,axis=1,keepdims=True)
    # fea_mfcc = fea_mfcc-

    file_fea = os.path.join(fea_train_path,spk+"_"+utt+".npy")
    np.save(file=file_fea,arr=fea_mfcc)
    print("save_file ",file_fea)

time = time / 400
w = "time_average : " + str(time) + '\n'
f.seek(0, 2)
f.write(w)
f.close()
# 提取 test数据特征

fea_train_path = "fea/test"
os.makedirs(fea_train_path,exist_ok=True)

file_lines = np.loadtxt('test.scp',dtype='str',delimiter=" ")
files= file_lines[:,0]
spk_ids = file_lines[:,1]
utt_ids = file_lines[:,2]

for file,spk,utt in zip(files,spk_ids,utt_ids):
    # 读取音频文件
    y,fs = librosa.load(file,sr=None, mono=True)

    # 进行MFCC特征的提取
    raw_mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=19, n_fft=512, hop_length=160, win_length=320, n_mels=20)
#    raw_mfccs = librosa.feature.mfcc(y=y, sr=sr, S=None, n_mfcc=3, dct_type=2, norm='ortho')
    # 增加动态特征
    raw_mfcc = librosa.util.normalize(raw_mfcc)
    mfcc_delta = librosa.feature.delta(raw_mfcc)
    mfcc_delta2 = librosa.feature.delta(raw_mfcc, order=2)

    # 拼接生成最终的MFCC特征
    fea_mfcc = np.concatenate([raw_mfcc,mfcc_delta,mfcc_delta2],axis=0)
    file_fea = os.path.join(fea_train_path,spk+"_"+utt+".npy")
    np.save(file=file_fea,arr=fea_mfcc)
    print("save_file ",file_fea)

