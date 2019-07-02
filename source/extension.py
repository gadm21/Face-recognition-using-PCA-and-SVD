import cv2
import numpy as np
import os
# Target Width and Height of the face photo
W, H = 200, 200


# Target imgs folder, pre-processed imgs, "result" folder to save intermediate results
imgs_dir, res_dir = 'images', 'result'
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface.xml')

def capture_keys_and_save(counter, new_image):
    code= cv2.waitKey(1)
    if code == ord('p'):
        if new_image is not None:
            img_name = 'frame'+ str(counter)+ '.jpg'
            img_path = os.path.join(imgs_dir, img_name)
            cv2.imwrite(img_path, new_image)
            print "saved"
            return 3
        else: print "none"
    if code == ord('q'):
        return 1
    if code == ord('z'):
        return 2
    return 0

def draw_rectangles(img, rectangles):
    for (x, y, w, h) in rectangles:
        pt1, pt2 = (x, y), (x + w, y + h)
        cv2.rectangle(img, pt1, pt2, color=(0, 255, 0))
        
def calculate_area(rect):
    ''' Caclulate the area of the region (e.g. detected face) '''
    x, y, w, h = rect
    area = w * h
    return area
    
def find_larget_face(detected_faces):
    ''' Find the largest face among all detected faces '''
    # No faces found
    if len(detected_faces) == 0:
        print 'No faces found!'
        return None

    areas = [calculate_area(face) for face in detected_faces]
    max_index = np.argmax(areas)
    largest_face = detected_faces[max_index]
    return largest_face
    
def detect_face(image): 
    global faceCascade

    detected_face_resized, detected_face_coords = None, None
    detected_faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE)
    
    # Find the biggest face coordinates
    largest_info = find_larget_face(detected_faces)

    if largest_info is not None:
        space= 50
        (x, y, w, h) = largest_info
        
        detected_face_coords = largest_info  # copy location of largeest face (x,y,w,h)

        face_cropped = image[y:y + h, x:x + w]  # crop and fetch only face
        detected_face_resized = cv2.resize(face_cropped, (W, H))  # resize the largest face

    return detected_face_resized, detected_face_coords
    
def pre_processing():
    f_list = [f for f in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f))]
    
    for file_name in f_list:
        img_gray = cv2.imread(os.path.join(imgs_dir, file_name), 0)  # read image as grayscale
        cv2.imshow(file_name, img_gray)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

def compute_PCA():

    
    n = sum([True for f in f_list])  # Count total no of face images
    print "found:", str(n) , " faces"
    # compute the mean face
    mu = np.mean(X, 0)

    # Subtract the mean face from each image before performing SVD and PCA
    ma_data = X - mu
    
    U, S, Vt = np.linalg.svd(ma_data.transpose(), full_matrices=False)
    V = Vt.T
    

    #sorting according to S values (the biggest first)
    #ind = np.argsort(S)[::-1]
    #U, S, V = U[:, ind], S[ind], V[:, ind]
    e_faces = U
    
    
    # Weights is an n x n matrix
    weights = np.dot(ma_data, e_faces)  # TODO: Maybe swap + .T to e_faces

    
    print "X: ", X.shape
    print "mu: ", mu.shape
    print "eigenfaces: ", U.shape
    print "S: ", S.shape
    print "wights: ", weights.shape
    cv2.imwrite(os.path.join(res_dir, 'eigenfacce.jpg'), U[:, 0].reshape(im_width, im_height, 3))
    
    # Some intermediate save:
    save_mean_face = True
    if save_mean_face:
        # Save mean face
        print mu.size
        mean_face = mu.reshape(im_width, im_height, 3)
        cv2.imwrite(os.path.join(res_dir, 'mean_face.jpg'), mean_face)
        # plt.imshow(mean_face, cmap='gray'); plt.show()

    save_eigenvectors = False
    if save_eigenvectors:
        print("Writing eigenvectors to disk...")
        for i in xrange(n):
            f_name = os.path.join(res_dir, 'eigenvector_%s.png' % i)
            im = U[:, i].reshape(im_width, im_height, 3)
            cv2.imwrite(f_name, im)

    save_reconstructed = True
    if save_reconstructed:
        k = 151
        print '\n', 'Save the reconstructed images based on only "%s" eigenfaces' % k
        for img_id in range(n):
            # for k in range(1, total + 1):
            recon_img = mu + np.dot(weights[img_id, :k], e_faces[:, :k].T)
            recon_img.shape = (im_width, im_height, 3)  # transform vector to initial image size
            cv2.imwrite(os.path.join(res_dir, 'img_reconstr_%s_k=%s.png' % (f_list[img_id], k)), recon_img)
            
def recognize(image, flattened_mean):
    if image is None: return
    flattened_image= image.flatten()
    my_img = flattened_image - flattened_mean
    
    weights = np.dot(my_img, my_img) # my projection on eigen faces
    cv2.imshow("re", test_f.reshape(100, 100, 3))
    

    