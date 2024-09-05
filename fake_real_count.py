import os

def count_file(path):
    files = os.listdir(path)
    fake = 0
    real = 0
    for file in files:
        if file[0:4]=='fake':
            fake = fake + 1
        else:
            real = real + 1

    return fake,real