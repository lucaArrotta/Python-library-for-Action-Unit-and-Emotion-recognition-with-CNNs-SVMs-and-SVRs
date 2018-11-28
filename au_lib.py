from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.externals import joblib
from PIL import Image
import operator
from face_helpers import get_face_locations, get_landmarks, align_and_crop, get_hog_features, show_gray_image, show_image_from_path


def apply_pipeline(path, return_hog_image=False):
    img = get_face_locations(path, verbose=False)
    landmarks, _ = get_landmarks(img)
    img = align_and_crop(img, landmarks)
    return get_hog_features(img, flat_vector=True, return_image=return_hog_image)


models_path = "models"
def save_model(model, name):
    joblib.dump(model, "{}/{}".format(models_path, name))
    
def load_model(name):
    return joblib.load("{}/{}".format(models_path, name))


models_dict = {"svm1": load_model("svm_au1"), "svm2": load_model("svm_au2"), "svm4": load_model("svm_au4"), "svm5": load_model("svm_au5"), "svm6": load_model("svm_au6"), "svm7": load_model("svm_au7"), "svm9": load_model("svm_au9"), "svm12": load_model("svm_au12"), "svm14": load_model("svm_au14"), "svm15": load_model("svm_au15"), "svm16": load_model("svm_au16"), "svm20": load_model("svm_au20"), "svm23": load_model("svm_au23"), "svm26": load_model("svm_au26"),                "pca1": load_model("pca_au1"), "pca2": load_model("pca_au2"), "pca4": load_model("pca_au4"), "pca5": load_model("pca_au5"), "pca6": load_model("pca_au6"), "pca7": load_model("pca_au7"), "pca9": load_model("pca_au9"), "pca12": load_model("pca_au12"), "pca14": load_model("pca_au14"), "pca15": load_model("pca_au15"), "pca16": load_model("pca_au16"), "pca20": load_model("pca_au20"), "pca23": load_model("pca_au23"), "pca26": load_model("pca_au26")}
aus = [1, 2, 4, 5, 6, 7, 9, 12, 14, 15, 16, 20, 23, 26]


# funziona sia passandogli un path che un'immagine direttamente
def get_img_aus(img):
    global aus
    global models_dict
    img_aus = list()
    orig_fd = apply_pipeline(img)
    for au in aus:
        pca = models_dict["pca{}".format(au)]
        clf = models_dict["svm{}".format(au)]
        fd = pca.transform(orig_fd.reshape(1,-1))
        if clf.predict(fd)[0] == 1:
            img_aus.append(au)
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


def get_img_emotions(img):
    emotions = dict()
    img_aus = get_img_aus(img)
    emotions["happiness"] = happinessMean(img_aus)
    emotions["sadness"] = sadnessMean(img_aus)
    emotions["surprise"] = surpriseMean(img_aus)
    emotions["fear"] = fearMean(img_aus)
    emotions["anger"] = angerMean(img_aus)
    emotions["disgust"] = disgustMean(img_aus)
    return emotions


def get_img_relative_emotions(img):
    emotions = get_img_emotions(img)
    s = sum(emotions.values())
    for k, v in emotions.items():
        pct = round(min(v * 100.00 / s, v*100.00), 2)  # il secondo operatore del min l'ho aggiunto. per avere sempre percentuali non va bene il min
        emotions[k] = pct
    return emotions
        


def get_maximum_emotion(img):
    emotions = get_img_emotions(img)
    return max(emotions.items(), key=operator.itemgetter(1))


