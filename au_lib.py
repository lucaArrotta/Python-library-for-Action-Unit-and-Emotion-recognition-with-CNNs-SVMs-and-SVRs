from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib
from PIL import Image
import numpy as np
import operator
from face_helpers import get_face_locations, get_landmarks, align_and_crop, get_hog_features, show_gray_image, show_image_from_path


def get_flatten_landmarks(landmarks):
    # From the landmarks list (which contains a dict with the features) we obtain a flat array
    lista_landmarks = list()
    for el in landmarks[0].values():
        lista_landmarks.append(el)
    return np.asarray([el for sublist in lista_landmarks for item in sublist for el in item])
    

def apply_pipeline(path, return_hog_image=False):
    img = get_face_locations(path, verbose=False)
    landmarks, _ = get_landmarks(img)
    img = align_and_crop(img, landmarks)
    landmarks, _ = get_landmarks(img)
    if len(landmarks) == 0:  # it can't find landmarks, so we don't consider this example
        landmarks = []
    else:
        landmarks = get_flatten_landmarks(landmarks)
    return (get_hog_features(img, flat_vector=True, return_image=return_hog_image), landmarks)


models_path = "models"
def save_model(model, name):
    joblib.dump(model, "{}/{}".format(models_path, name))
    
def load_model(name):
    return joblib.load("{}/{}".format(models_path, name))


models_dict = {"svm1": load_model("svm_au1"), "svm2": load_model("svm_au2"), "svm4": load_model("svm_au4"), "svm5": load_model("svm_au5"), "svm6": load_model("svm_au6"), "svm7": load_model("svm_au7"), "svm9": load_model("svm_au9"), "svm12": load_model("svm_au12"), "svm14": load_model("svm_au14"), "svm15": load_model("svm_au15"), "svm16": load_model("svm_au16"), "svm20": load_model("svm_au20"), "svm23": load_model("svm_au23"), "svm26": load_model("svm_au26"), \
			   "pca1-svm": load_model("pca_au1-svm"), "pca2-svm": load_model("pca_au2-svm"), "pca4-svm": load_model("pca_au4-svm"), "pca5-svm": load_model("pca_au5-svm"), "pca6-svm": load_model("pca_au6-svm"), "pca7-svm": load_model("pca_au7-svm"), "pca9-svm": load_model("pca_au9-svm"), "pca12-svm": load_model("pca_au12-svm"), "pca14-svm": load_model("pca_au14-svm"), "pca15-svm": load_model("pca_au15-svm"), "pca16-svm": load_model("pca_au16-svm"), "pca20-svm": load_model("pca_au20-svm"), "pca23-svm": load_model("pca_au23-svm"), "pca26-svm": load_model("pca_au26-svm"), \
			   "scaler1-svm": load_model("scaler_au1-svm"), "scaler2-svm": load_model("scaler_au2-svm"), "scaler4-svm": load_model("scaler_au4-svm"), "scaler5-svm": load_model("scaler_au5-svm"), "scaler6-svm": load_model("scaler_au6-svm"), "scaler7-svm": load_model("scaler_au7-svm"), "scaler9-svm": load_model("scaler_au9-svm"), "scaler12-svm": load_model("scaler_au12-svm"), "scaler14-svm": load_model("scaler_au14-svm"), "scaler15-svm": load_model("scaler_au15-svm"), "scaler16-svm": load_model("scaler_au16-svm"), "scaler20-svm": load_model("scaler_au20-svm"), "scaler23-svm": load_model("scaler_au23-svm"), "scaler26-svm": load_model("scaler_au26-svm"), \
			   "svr1": load_model("svr_au1"), "svr2": load_model("svr_au2"), "svr4": load_model("svr_au4"), "svr5": load_model("svr_au5"), "svr6": load_model("svr_au6"), "svr7": load_model("svr_au7"), "svr9": load_model("svr_au9"), "svr12": load_model("svr_au12"), "svr14": load_model("svr_au14"), "svr15": load_model("svr_au15"), "svr16": load_model("svr_au16"), "svr20": load_model("svr_au20"), "svr23": load_model("svr_au23"), "svr26": load_model("svr_au26"), \
			   "pca1-svr": load_model("pca_au1-svr"), "pca2-svr": load_model("pca_au2-svr"), "pca4-svr": load_model("pca_au4-svr"), "pca5-svr": load_model("pca_au5-svr"), "pca6-svr": load_model("pca_au6-svr"), "pca7-svr": load_model("pca_au7-svr"), "pca9-svr": load_model("pca_au9-svr"), "pca12-svr": load_model("pca_au12-svr"), "pca14-svr": load_model("pca_au14-svr"), "pca15-svr": load_model("pca_au15-svr"), "pca16-svr": load_model("pca_au16-svr"), "pca20-svr": load_model("pca_au20-svr"), "pca23-svr": load_model("pca_au23-svr"), "pca26-svr": load_model("pca_au26-svr"), \
			   "scaler1-svr": load_model("scaler_au1-svr"), "scaler2-svr": load_model("scaler_au2-svr"), "scaler4-svr": load_model("scaler_au4-svr"), "scaler5-svr": load_model("scaler_au5-svr"), "scaler6-svr": load_model("scaler_au6-svr"), "scaler7-svr": load_model("scaler_au7-svr"), "scaler9-svr": load_model("scaler_au9-svr"), "scaler12-svr": load_model("scaler_au12-svr"), "scaler14-svr": load_model("scaler_au14-svr"), "scaler15-svr": load_model("scaler_au15-svr"), "scaler16-svr": load_model("scaler_au16-svr"), "scaler20-svr": load_model("scaler_au20-svr"), "scaler23-svr": load_model("scaler_au23-svr"), "scaler26-svr": load_model("scaler_au26-svr")}


aus = [1, 2, 4, 5, 6, 7, 9, 12, 14, 15, 16, 20, 23, 26]


# funziona sia passandogli un path che un'immagine direttamente
def get_img_aus_occurencies(img):
    global aus
    global models_dict
    img_aus = list()
    orig_fd, landmarks = apply_pipeline(img)
    if len(landmarks) > 0:
	    for au in aus:
	        pca = models_dict["pca{}-svm".format(au)]
	        scaler = models_dict["scaler{}-svm".format(au)]
	        clf = models_dict["svm{}".format(au)]
	        fd = pca.transform(orig_fd.reshape(1,-1))
	        fd = np.concatenate((fd[0,:], landmarks)).reshape(1,-1)
	        fd = scaler.transform(fd)
	        if clf.predict(fd)[0] == 1:
	            img_aus.append(au)
    return img_aus


def get_img_aus_intensities(img):
    global aus
    global models_dict
    img_aus = dict()
    orig_fd, landmarks = apply_pipeline(img)
    if len(landmarks) > 0:
	    for au in aus:
	        pca = models_dict["pca{}-svr".format(au)]
	        scaler = models_dict["scaler{}-svr".format(au)]
	        clf = models_dict["svr{}".format(au)]
	        fd = pca.transform(orig_fd.reshape(1,-1))
	        fd = np.concatenate((fd[0,:], landmarks)).reshape(1,-1)
	        fd = scaler.transform(fd)
	        pred = clf.predict(fd)[0]
	        if int(pred) > 0:
	        	img_aus[au] = round(pred, 2)
    return img_aus


def happinessMean(aus):
    happiness_aus = [6, 12]
    return round(sum([aus.count(x) for x in happiness_aus])/len(happiness_aus), 2)


def sadnessMean(aus):
    happiness_aus = [1, 4, 15]
    return round(sum([aus.count(x) for x in happiness_aus])/len(happiness_aus), 2)


def surpriseMean(aus):
    happiness_aus = [1, 2, 5, 26]
    return round(sum([aus.count(x) for x in happiness_aus])/len(happiness_aus), 2)


def fearMean(aus):
    happiness_aus = [1, 2, 4, 5, 7, 20, 26]
    return round(sum([aus.count(x) for x in happiness_aus])/len(happiness_aus), 2)


def angerMean(aus):
    happiness_aus = [4, 5, 7, 23]
    return round(sum([aus.count(x) for x in happiness_aus])/len(happiness_aus), 2)


def disgustMean(aus):
    happiness_aus = [9, 15, 16]
    return round(sum([aus.count(x) for x in happiness_aus])/len(happiness_aus), 2)


def get_img_emotions_occurencies(img):
    emotions = dict()
    img_aus = get_img_aus_occurencies(img)
    emotions["happiness"] = happinessMean(img_aus)
    emotions["sadness"] = sadnessMean(img_aus)
    emotions["surprise"] = surpriseMean(img_aus)
    emotions["fear"] = fearMean(img_aus)
    emotions["anger"] = angerMean(img_aus)
    emotions["disgust"] = disgustMean(img_aus)
    return emotions


'''
def get_img_relative_emotions(img):
    emotions = get_img_emotions(img)
    s = sum(emotions.values())
    for k, v in emotions.items():
        pct = round(min(v * 100.00 / s, v*100.00), 2)
        emotions[k] = pct
    return emotions
        


def get_maximum_emotion(img):
    emotions = get_img_emotions(img)
    return max(emotions.items(), key=operator.itemgetter(1))
'''


