import os
import shutil

source = 'test/mimic/'

for dir in os.listdir(source):
    for im in os.listdir(source + '/' + dir):
        shutil.copy(source + '/' + dir + '/' + im, 'test/mimic/' + im)

source = 'test/pad/'

for dir in os.listdir(source):
    for im in os.listdir(source + '/' + dir):
        shutil.copy(source + '/' + dir + '/' + im, 'test/pad/' + im)

source = 'test/cxr/'

for dir in os.listdir(source):
    for im in os.listdir(source + '/' + dir):
        shutil.copy(source + '/' + dir + '/' + im, 'test/cxr/' + im)
