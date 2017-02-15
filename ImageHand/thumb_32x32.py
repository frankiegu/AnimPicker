# encoding=utf-8
from PIL import Image
from pylab import *
import os
import imghdr
import sys

count = 0


def readFolder(file_path):
	pathDir = os.listdir(file_path)

	save_path = file_path + "_thumb"
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	for name in pathDir:
		fileName = os.path.join(file_path, name)
		if imghdr.what(fileName):
			minImage(file_path, name, save_path)
		if count > 0 and count % 100 == 0:
			print "已经处理" + str(count) + "张图片"

	print "一共处理" + str(count) + "张图片"

def minImage(file_path, name, save_path):
	global count
	im = Image.open(os.path.join(file_path, name)).convert('RGB')
	new_im = im.crop(get_crop_size(im))
	new_im.thumbnail((32,32))
	new_im.save(os.path.join(save_path, name))
	count += 1

def get_crop_size(im):
	x, y = im.size
	if x > y:
		d = (x - y) / 2
		return (d, 0, d+y, y)
	else:
		d = (y - x) / 2
		return (0, d, x, x+d)

def main(argv):
	file_path = argv
	readFolder(file_path)

if __name__ == '__main__':
	main(sys.argv[1])