import os

def set_label(path):
    files = os.listdir(path)

    labels = []

    for file in files:
        labels.append(file[0:4])
    
    return labels


