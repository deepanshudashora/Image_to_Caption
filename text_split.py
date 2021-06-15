# For Text description
import os
test_images = os.listdir("Data/testing_images")


f_train=open('Data/training_captions.txt')  
f_test=open('Data/testing_caption.txt','a')

def remove_line(file,x):
    with open(file, "r+") as f:
        d = f.readlines()
        f.seek(0)
        for i in d:
            if i != x:
                f.write(i)
        f.truncate()


for x in f_train.readlines():
    if "," in x[0:24]:
        if x[0:23] in test_images:
            f_test.write(x)
            remove_line("Data/training_captions.txt",x)
    else:
        if x[0:24] in test_images:
            f_test.write(x)
            remove_line("Data/training_captions.txt",x)
f_train.close()
f_test.close()