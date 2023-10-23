import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()

images_p1 = ["personas/LPavez1.jpg", "personas/LPavez2.jpeg", "personas/LPavez3.jpeg","LPavez4.jpeg"]
images_p2 = ["personas/rodrigograf/rg1.jpg", "personas/rodrigograf/rg2.jpg", "personas/rodrigograf/rg3.jpg", "personas/rodrigograf/rg4.jpg", "personas/rodrigograf/rg5.jpg"]
images_p3 = ["personas/alanlahaye/al1.jpeg", "personas/alanlahaye/al2.jpeg", "personas/alanlahaye/al3.jpeg", "personas/alanlahaye/al4.jpeg", "personas/alanlahaye/al5.jpeg"]
images_p4 = ["personas/diegoheredia/dh1.jpeg", "personas/diegoheredia/dh2.jpeg", "personas/diegoheredia/dh3.jpeg", "personas/diegoheredia/dh4.jpeg", "personas/diegoheredia/dh5.jpeg"]
images_p5 = ["personas/franfernandoy/ff1.jpeg", "personas/franfernandoy/ff2.jpeg", "personas/franfernandoy/ff3.jpeg", "personas/franfernandoy/ff4.jpeg", "personas/franfernandoy/ff5.jpeg"]

embeddings = []
i=0
for p in [images_p1, images_p2, images_p3, images_p4, images_p5]:
    embeddings_person = []
    for image in p:
        img = cv2.imread(image)
        faces = app.get(img)
        embeddings_person.append(faces[0]['embedding'])
    nparray = np.array(embeddings_person)
    nparray = np.stack(nparray)
    nparray = np.average(nparray, axis=0)
    embeddings.append(nparray)
    np.save(f"embeddings/_persona{i}", nparray)
    i+=1
