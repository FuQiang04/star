import os

if __name__ == "__main__":
    print("请输入录音人数 : ", end='')
    num_people = eval(input())
    for i in range(num_people):
        print("请输入第{}位说话人姓名 : ".format(str(i + 1)), end='')
        os.system('python generate_spk_GMM.py')
