import os


if __name__ == "__main__":
    for i in range(1, 10):
        os.system('python remove.py')
#    os.system('python gen_ubm_scp.py')
        os.system('python gen_enrollment_scp.py')
        os.system('python feature_extract.py')
#    os.system('python train_UBM.py')
        os.system('python train_spk_model.py')
        os.system('python eval_score.py')
