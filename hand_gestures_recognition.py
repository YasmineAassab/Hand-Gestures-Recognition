
# importer les packages nécessaires pour le projet
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model

#----------------------------------initialisation de MediaPipe

#Exécuter l'algorithme de reconnaissance de la main
mpHands = mp.solutions.hands
#Configurer le modèle (détecter une seule main)
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
#dessinez les points clés détectés pour la main
mpDraw = mp.solutions.drawing_utils

#---------------------------------- initialisation de TensorFlow

#Charger le modèle de reconnaissance de geste pré-entraîné par TensorFlow
model = load_model('mp_hand_gesture')

#---------------------------------- Charger les noms de classe
# Le fichier gesture.names contient le nom des classes de mouvements.
f = open('gestures.names', 'r')
classNames = f.read().split('\n')  # lire le fichier en utilisant la fonction read().
f.close()
#print(classNames)


# #---------------------------------- Initialisation du caméra the webcam
cap = cv2.VideoCapture(0)

while True:
    # Read each frame from the webcam
    _, frame = cap.read()
    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)

    #Convertir le format des images RGB en format RVB
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Obtenir la prédiction du point de repère de la main
    result = hands.process(framergb)

    className = ''

#---------------------------------------------traiter le résultat
    #Vérifier si une main est détectée
    if result.multi_hand_landmarks:
        landmarks = []
        #Parcourir chaque détection
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                #Stocker les coordonnées sur une liste
                landmarks.append([lmx, lmy])

            #Dessiner des repères sur des cadres
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]


    # show the prediction on the frame
    cv2.putText(frame, className, (50, 50), cv2.FONT_HERSHEY_DUPLEX,
                   1, (0,0,0), 2 , cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Hand gesture", frame)

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()