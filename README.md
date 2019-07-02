# Face-recognition-using-PCA-and-SVD

## requirements
- python 2.7
- cv2
- os
- numpy




## process
1. put `main.py`, `extension.py`, and your dataset of face images in a folder named `images` all in one folder.


2. the first line in the `main` function in the `main.py` file is:```python mode= True ```

    there are two modes of operation determined by that `mode` variable:

   * if set to `True`: the program will detect the face infront of the camera and, automatically, open another window containing the image in the dataset of the face closest to the face infront of the camera.
    ![](https://github.com/gadm21/Face-recognition-using-PCA-and-SVD/imgs/mode_true.PNG)
   * if set to `False`: the program will detect the face infront of the camera and whenever the key `p` is pressed, the detected face (i.e. the image inside the green box) will be saved to the `images` folder.
    ![](https://github.com/gadm21/Face-recognition-using-PCA-and-SVD/imgs/mode_false.PNG)

3. Run the `main.py` file


4. press `q` to quit

 
