import os

path1 = "fea/test"
path2 = "fea/train"
path3 = "../DataSource/models"

for path, names, filenames in os.walk(path1):
    for file_path in filenames:
        os.remove(path1 + '/' + file_path)
for path, names, filenames in os.walk(path2):
    for file_path in filenames:
        os.remove(path2 + '/' + file_path)
for path, names, filenames in os.walk(path3):
    for file_path in filenames:
        if file_path == "ubm.model":
            continue
        os.remove(path3 + '/' + file_path)
if os.path.exists("test.scp"):
    os.remove("test.scp")
if os.path.exists("train.scp"):
    os.remove("train.scp")
