from au_lib import get_img_aus_occurencies, get_img_emotions_occurencies, get_img_aus_intensities
from PIL import Image
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
	aus = get_img_aus_occurencies(sys.argv[1])
	emotions = get_img_emotions_occurencies(sys.argv[1])
	aus_int = get_img_aus_intensities(sys.argv[1])
	#print("AUs occurencies:  ", aus)
	#print("emotions: ", emotions)
	#print("AUs intensities:  ", aus_int)
	img = Image.open(sys.argv[1])
	imgplot = plt.imshow(img)
	plt.title("AUs (occurencies): {}\n{}\nAUs (intensities): {}".format(aus, emotions, aus_int))
	plt.show()

