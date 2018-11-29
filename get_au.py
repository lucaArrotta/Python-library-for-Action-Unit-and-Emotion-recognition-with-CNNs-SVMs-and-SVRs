from au_lib import get_img_aus_occurencies, get_img_emotions_occurencies, get_img_aus_intensities, get_img_emotions_intensities
from PIL import Image
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
	if sys.argv[2] == "--occurencies":
		aus = get_img_aus_occurencies(sys.argv[1])
		emotions = get_img_emotions_occurencies(sys.argv[1])
	if sys.argv[2] == "--intensities":
		aus = get_img_aus_intensities(sys.argv[1])
		emotions = get_img_emotions_intensities(sys.argv[1])
	print("aus:  ", aus)
	print("emotions: ", emotions)
	img = Image.open(sys.argv[1])
	imgplot = plt.imshow(img)
	plt.title("AUs: {}\n{}".format(aus, emotions))
	plt.show()

