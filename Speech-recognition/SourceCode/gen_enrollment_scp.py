import  os
import numpy as np
import random
# 生成train.scp和test.scp
root_path= "../DataSource/source-train"
#print("input(for train 13 to 1): ", end='')
#num = eval(input())
f = open("log.txt", 'a')
f.seek(0, 2)
for i in range(1,10):
    k = str(i) + '.txt'
    if not os.path.exists(k):
        num = i
        w = "train_num : " + str(num) + '\n'
        f.write(w)
        print("oooooooooooo")
        f.close()
        f1 = open(k, 'w')
        f1.close()
        break

f_train = open('train.scp', 'wt')
with open("test.scp",'wt') as f_test:
    for dirpath, dirnames, filenames in os.walk(root_path):
        L = random.sample(range(1, 14), num)
        for file_path in filenames:
            if file_path.endswith(".wav"):
                full_name= os.path.join(dirpath,file_path)
                speak_id = os.path.split(dirpath)[-1]
                utt_id = file_path.split(".")[0]
                if utt_id[-2] == '0':
                    k = utt_id[-1]
                else:
                    k = utt_id[-2:]
                if eval(k) in L:
                    f_train.write("%s %s %s\n" % (full_name, speak_id, utt_id))
                else:
                    f_test.write("%s %s %s\n"%(full_name,speak_id,utt_id))
                print("%s %s %s"%(full_name,speak_id,utt_id))
f_train.close()


