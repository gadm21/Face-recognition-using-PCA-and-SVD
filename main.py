#from training import *
import cv2
import os 
import numpy as np
from extension import *






def get_data_matrix(images_location):
    
    #makeing a list of images names from directory imgs_dir
    f_list = [f_name for f_name in os.listdir(images_location) if os.path.isfile(os.path.join(images_location, f_name))]
    
    #opening the images and resizeing it to (W*H*3) and flattening them to rows and finally storing them in a matrix
    A = np.array([cv2.resize(cv2.imread(os.path.join(images_location, filename), -1), (W,H)).flatten() for filename in f_list])
    
    #calculating the images size by getting the image of any image
    image_size= cv2.resize(cv2.imread(os.path.join(images_location, f_list[0]), -1), (W,H)).shape
    #image_size= (W, H, 3)
    
    return image_size, A






def reshaping(mean, eigen_vectors, image_size):
    mean_face= mean.reshape(image_size)
    
    eigen_faces= []
    for eigen_vector in eigen_vectors:
        eigen_face= eigen_vector.reshape(image_size)
        eigen_faces.append(eigen_face)
        
    return mean_face, eigen_faces







def find_distance(w1, w2):
    return np.sqrt(np.sum((w1 - w2)** 2)) 
    







#def construct_face(mean_face, eigen_faces):
def get_image_by_id(index):
    f_list = [f_name for f_name in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f_name))]
    if index >= len(f_list):
        return False
    
    target_image= cv2.resize(cv2.imread(os.path.join(imgs_dir, f_list[index]), -1), (W,H))
    
    return target_image
    
    
    
def calculate_weights(mean, A, eigen_vectors):
    data_variances= A- mean
    weights= np.dot(data_variances, eigen_vectors.T)
    print A.shape
    print eigen_vectors.shape
    print weights.shape
    
    return weights
   
    
    
    
    
    
    
    
def find_closest_image(mean, eigen_vectors, weights, frame):
    flattened_frame= frame.flatten()
    
    relocated_frame= flattened_frame- mean
    frame_projection= np.dot(relocated_frame, eigen_vectors.T)
    print "shape: ", relocated_frame.shape
    print "shape: ", eigen_vectors.shape
    f_list = [f_name for f_name in os.listdir(imgs_dir) if os.path.isfile(os.path.join(imgs_dir, f_name))]
    images_num = sum([True for f in f_list])  # Count total no of face images    
    
    distances= []
    current_low= 0
    for image_index in range(images_num):
        distance= find_distance(weights[image_index], frame_projection)
        distances.append(distance)
        if image_index> 0:
            if distances[current_low]> distances[image_index]:
                current_low= image_index
    
    closest_image= get_image_by_id(current_low)
    
    return current_low, closest_image
    
    
    
    
    
    
    
    
    
    
#.................................................................................................................................
#.................................................................................................................................
#.................................................................................................................................
#.................................................................................................................................
#.................................................................................................................................
#.................................................................................................................................
def main():
    
    mode= True  #set if you want to recognize faces based on the available dataset, unset if you want to capture more faces
    
    image_size, A= get_data_matrix(imgs_dir)
    
    #calculating eigen_vector and mean using PCA
    components= (W*H*3)/10
    print "c:", components
    mean, eigen_vectors = cv2.PCACompute(A, mean=None ,maxComponents=components)
    mean_face, eigen_faces= reshaping(mean, eigen_vectors, image_size)
    #print mean_face.shape
    weights= calculate_weights(mean, A, eigen_vectors)
    
    cv2.imwrite("re.jpg", mean_face)
    cv2.waitKey(0)
    
    
    cap = cv2.VideoCapture(0)
    counter=0
    while(cap.isOpened()):
        counter+=1
        
        ret, frame = cap.read()  # Capture frame-by-frame
        # = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        detected_face, detected_face_coords = detect_face(frame)  # try to detect face
    
        if detected_face is not None:
            draw_rectangles(frame, [detected_face_coords])

        
        new_image= detected_face if detected_face is not None else frame
        show_recognized= True
        if detected_face is not None and show_recognized:
            #cv2.namedWindow("Result", cv2.WINDOW_AUTOSIZE)
            (x, y, w, h)= detected_face_coords
            new_image=  frame[y:y+h, x:x+w] if detected_face is not None else frame
            resized_new_image= cv2.resize( new_image, (W,H))
            if mode:
                index, image= find_closest_image(mean, eigen_vectors, weights, resized_new_image)
                cv2.imshow("Result", image)
        
        cv2.imshow('Video', frame)  # show either frame (if face isn't detected or frame with mask)
        
        reply= capture_keys_and_save(counter, new_image)
        if reply==1: break
        elif reply==2: recognize(new_image, mean_face_flatten)
            
    cap.release()
    cv2.destroyAllWindows()
    
if __name__== "__main__":
    #compute_PCA()
    main()
    