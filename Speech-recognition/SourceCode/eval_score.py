import numpy as np
import os
import joblib
import sklearn


if __name__ == "__main__":
    
    # 加载UBM
    path_model = '../DataSource/models'
    ubm = joblib.load(os.path.join(path_model,'ubm.model'))
    
    # 加载验证数据
    path_fea= 'fea/test'
    file_lines = np.loadtxt("test.scp",dtype='str',delimiter=" ")
    test_file = file_lines[:,2]
    highest = []
    n = 0
    m = 0

    for test_file_n in test_file:
        file_fea = os.path.join(path_fea, test_file_n[:5] + '_' + test_file_n + '.npy')
        data = np.load(file_fea).T
        for root, dirs, files in os.walk(path_model):
            score_highest = 0
            for model in files:
                gmm = joblib.load(os.path.join(path_model,model))
                score_gmm = gmm.score(data)
                if score_gmm > score_highest:
                    spk = model[:-6]
                    score_highest = score_gmm
        highest.append(spk)
        print(test_file_n[:5],end=' ')
        print(spk)
        m += 1
        if(test_file_n[:5] == spk):
            n += 1
    accuracy = n/m
    f = open("log.txt", 'a')
    f.seek(0, 2)
    f.write("accuracy = {}\n\n".format(accuracy))
    print("accuracy = {}".format(accuracy))





