2  运行 gen_ubm_scp.py 生成 ubm_wav.scp
3  运行  gen_enrollment_scp.py  生成 test.scp, train.scp
4  运行 feature_extract.py  生成特征
5  运行 train_UBM.py 训练UBM
6  运行 train_spk_model.py 生成说话人GMM
7  运行 eval_score.py 进行测试打分计算EER

PyAudio == 0.2.14
joblib == 1.4.0
librosa == 0.10.1
numpy == 1.26.4
pydub == 0.25.1
scikit-learn == 1.4.2
setuptools == 69.5.1


