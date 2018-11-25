
# coding: utf-8

# # Modelli di Computazione Affettiva
# ### Luca Arrotta
# 
# ### Funzioni di Supporto
# Lo scopo di questo notebook è quello di implementare alcune funzioni di supporto.
# I nostri dataset presentano gli esempi (x,y) in questa forma:
# * x: singolo frame estratto da un video
# * y: etichetta associata al frame che contiene le Action Unit attive in quel frame
# 
# <br>
# 
# Lo scopo è quello di applicare una pipeline di operazioni sui singoli frame:
#     1. Rilevamento del volto del soggetto all'interno del frame
#     2. Rilevamento dei landmark di tale volto
#     3. Face alignment e maschera
#     4. Estrazione degli HOG
# In questo notebook sono implementate le funzioni di supporto necessarie per applicare queste 4 operazioni.
# 
# <br>
# 
# In un altro notebook, passeremo le feature degli HOG di ogni esempio del Training Set ad un algoritmo PCA per la riduzione della dimensiontalità.
# Alla fine otterremo un dataset in cui gli esempi (x,y) saranno:
# * x: dati relativi al singolo frame dopo l'applicazione della pipeline descritta
# * y: etichetta associata al frame che contiene le Action Unit attive in quel frame
#    
# 
# ***

# ### 1. Rilevamento del volto del soggetto
# Stampa di una delle immagini del dataset

# In[9]:


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def show_image(img):
    imgplot = plt.imshow(img)
    plt.show()

def show_image_from_path(path):
    img = mpimg.imread(path)
    imgplot = plt.imshow(img)
    plt.show()

show_image_from_path('obama2.jpeg')


# Rilevamento del viso del soggetto utilizzando la libreria [Face Recognition](https://github.com/ageitgey/face_recognition)

# In[10]:


import dlib
import face_recognition
from PIL import Image

def get_face_locations(path, verbose=True):
    # Carica l'immagine in un array numpy
    img = face_recognition.load_image_file(path)

    # Trova tutte le facce nell'immagine usando il modello di default (basato sulle HOG)
    face_locations = face_recognition.face_locations(img)
    if verbose:
        print("Ho trovato un numero di facce pari a {}.".format(len(face_locations)))

    for face_location in face_locations:
        # Stampa la posizione di ogni faccia che ha trovato
        top, right, bottom, left = face_location
        if verbose:
            print("Una faccia si trova qui:   Top: {}, Left: {}, Bottom: {}, Right: {}".format(top, left, bottom, right))

        face_image = img[top:bottom, left:right]
        out_image = Image.fromarray(face_image)
    return out_image   # ritorna l'ultima faccia che ha trovato
        
        
face_image = get_face_locations("obama2.jpeg", verbose=True)
show_image(face_image)


# ***

# ### 2. Estrazione dei Landmark
# Estrazione dei landmark utilizzando nuovamente [Face Recognition](https://github.com/ageitgey/face_recognition)

# In[11]:


from PIL import Image, ImageDraw
import face_recognition
import numpy

def get_landmarks(img):
    # Carichiamo l'immagine che contiene solo la faccia in un array numpy
    img = numpy.array(img)

    # Trova tutti i landmark della faccia
    landmarks = face_recognition.face_landmarks(img)

    # Creiamo un'immagine in cui andremo a disegnare i landmark
    pil_image_drawn = Image.fromarray(img)
    d = ImageDraw.Draw(pil_image_drawn)

    for face_landmarks in landmarks:
        # Stampa la posizione di ogni landmark
        #for facial_feature in face_landmarks.keys():
        #    print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))

        # Disegna ogni landmark
        for facial_feature in face_landmarks.keys():
            d.line(face_landmarks[facial_feature], fill='white', width=3)

    # Ritorniamo la lista dei landmark
    return (landmarks, pil_image_drawn)


(face_landmarks_list, image_drawn) = get_landmarks(face_image)
show_image(image_drawn)


# ***

# ### 3. Face Alignment e maschera
# Codice adattato da questo [tutorial](https://www.pyimagesearch.com/2017/05/22/face-alignment-with-opencv-and-python/)

# In[12]:


import FaceAligner
import cv2
import numpy as np

# Instanziamo un oggetto che ci permette di allineare la faccia
fa = FaceAligner.FaceAligner()


def translate_eyebrow_landmarks(landmarks):
    new_left_eyebrow = [(tup[0], tup[1]-15) for tup in landmarks[0]['left_eyebrow']]
    new_right_eyebrow = [(tup[0], tup[1]-15) for tup in landmarks[0]['right_eyebrow']]
    landmarks[0]['left_eyebrow'] = new_left_eyebrow
    landmarks[0]['right_eyebrow'] = new_right_eyebrow


def get_aligned_face(img, landmarks):
    # Dalla lista dei landmarks (che contiene un dizionario con le diverse feature) otteniamo un flat array
    lista_landmarks = list()
    translate_eyebrow_landmarks(landmarks)  # serve per mostrare un pezzo in più di fronte
    for el in landmarks[0].values():
        lista_landmarks.append(el)
    lista_landmarks = [item for sublist in lista_landmarks for item in sublist]

    # Carichiamo l'immagine (image) e la lista dei landmark (shape) come array numpy
    img = np.asarray(img)
    shape = np.asarray(lista_landmarks)

    # Inizializziamo una maschera
    remapped_shape = np.zeros_like(shape) 
    feature_mask = np.zeros((img.shape[0], img.shape[1]))   

    # Dall'immagine estraiamo solo la parte della faccia coi landmark
    remapped_shape = cv2.convexHull(shape)
    cv2.fillConvexPoly(feature_mask, remapped_shape[0:27], 1)
    feature_mask = feature_mask.astype(np.bool)

    # Costruiamo l'immagine di output...
    out_face = np.zeros_like(img)
    out_face[feature_mask] = np.asarray(img)[feature_mask]
    
    # ...e la allineiamo
    out_face = fa.align(out_face, lista_landmarks, desiredFaceWidth=img.shape[0])
    return out_face



def crop_image(img, tol=0):
    # Maschera per i pixel con valore maggiore della tolleranza
    mask = img > tol

    # Posizione di questi pixel
    coords = np.argwhere(mask)

    # Bounding box che contiene questi pixel
    minCoords = coords.min(axis=0)
    x0 = minCoords[0]
    y0 = minCoords[1]
    maxCoords = coords.max(axis=0) + 1  # slices are exclusive at the top
    x1 = maxCoords[0]   
    y1 = maxCoords[1]

    # Prendo il contenuto della Bounding Box
    cropped = img[x0:x1, y0:y1]
    return cropped

def align_and_crop(img, landmarks, tol=0):
    img = get_aligned_face(img, landmarks)
    img = crop_image(img)
    img = np.asarray(Image.fromarray(img).resize((112,112)))
    return img


# In[13]:


#out_face = get_aligned_face(face_image, face_landmarks_list)
#out_face = crop_image(out_face)
#out_face = np.asarray(Image.fromarray(out_face).resize((112,112)))

out_face = align_and_crop(face_image, face_landmarks_list)

print("{} x {}".format(out_face.shape[0], out_face.shape[1]))
show_image(out_face)

(face_landmarks_list, image_drawn) = get_landmarks(out_face)  # ricalcolo i landmark su questo nuova immagine 112x112
#show_image(image_drawn)


# ***

# ### 4. Estrazione degli HOG
# Utilizzando la libreria [scikit-image](http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)

# In[29]:


from skimage.feature import hog
from skimage import data, exposure

def show_gray_image(img):
    imgplot = plt.imshow(img, cmap=plt.cm.gray)
    plt.show()

def get_hog_features(img, flat_vector=True, return_image=False):
    if return_image:
        fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=return_image, 
                    multichannel=True, block_norm='L2-Hys', feature_vector=flat_vector)
        # Riscaliamo le intensità per una visualizzazione migliore
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 15))
        return fd, hog_image_rescaled
    else:
        fd = hog(img, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=return_image, 
                    multichannel=True, block_norm='L2-Hys', feature_vector=flat_vector)
        return fd
    
features, hog_image = get_hog_features(out_face, flat_vector=True, return_image=True)
print(features.shape)
show_gray_image(hog_image)


# ***
