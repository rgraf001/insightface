import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

embeddings = []
personas = ["_persona0", "_persona1", "_persona2", "_persona3", "_persona4"]
for nombre in personas:
    embeddings.append(np.load(f"embeddings/{nombre}.npy"))

import cv2 as cv


def nombre_persona(_personaX):
    # Crear un diccionario con las asociaciones entre las variables y los nombres
    nombres = {
        "_persona0": "Luis",
        "_persona1": "Rodrigo",
        "_persona2": "Alan",
        "_persona3": "Diego",
        "_persona4": "Panchita BB"
    }

    if _personaX in nombres:
        nombre = nombres[_personaX]
    else:
        nombre = "no reconocido"
    return nombre

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    faces = app.get(frame)
    for face in faces:
        bbox = face["bbox"]
        frame = cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 1)
        x, y = int(bbox[0]), int(bbox[1])

        new_embedding = face["embedding"]
        from numpy import dot
        from numpy.linalg import norm

        for i, embedding in enumerate(embeddings):
            cos_sim = dot(embedding, new_embedding)/(norm(new_embedding)*norm(embedding))
            #print(f"persona {personas[i]}: similitud de {cos_sim * 100}%")
            print(100 * '-')
            if cos_sim >= 0.30:
                print ("Detectado: "+nombre_persona(personas[i]))
                print (f"probabilidad {cos_sim * 100}%")

                label = nombre_persona(personas[i])+f"{ round(cos_sim * 100 ,2)}%"
                cv2.putText(frame, label , (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    # Display the resulting frame
    cv.imshow('frame', frame)


    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
cap.release()
cv.destroyAllWindows()

# images_p1 = ["LPavez1.jpg"]
# images_p2 = ["LPavez1.jpg", "LPavez3.jpeg","LPavez4.jpeg"]

# embeddings = []
# personas = ["persona0", "persona1"]
# for nombre in personas:
#     embeddings.append(np.load(f"{nombre}.npy"))

# img = cv2.imread("LPavez1.jpg")

# faces = app.get(img)
# new_embedding = faces[0]["embedding"]

# from numpy import dot
# from numpy.linalg import norm

# for i, embedding in enumerate(embeddings):
#     cos_sim = dot(embedding, new_embedding)/(norm(new_embedding)*norm(embedding))
#     print(f"persona {personas[i]}: similitud de {cos_sim * 100}%")