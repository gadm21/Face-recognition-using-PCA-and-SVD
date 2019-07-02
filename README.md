# Face-recognition-using-PCA-and-SVD

## requirements
- python 2.7
- cv2
- os
- numpy




## process
1. put `main.py`, `extension.py`, `haarcascade_frontalface.xml`, and a folder named `images` containing the dataset all in one folder.


2. the first line in the `main` function in the `main.py` file is:```python mode= True ```

    there are two modes of operation determined by that `mode` variable:

   * if set to `True`: the program will detect the face infront of the camera and, automatically, open another window containing the image in the dataset of the face closest to the face infront of the camera.
    ![](https://github.com/gadm21/Face-recognition-using-PCA-and-SVD/blob/master/imgs/mode_true.PNG)
   * if set to `False`: the program will detect the face infront of the camera and whenever the key `p` is pressed, the detected face (i.e. the image inside the green box) will be saved to the `images` folder.
    ![](https://github.com/gadm21/Face-recognition-using-PCA-and-SVD/blob/master/imgs/mode_false.PNG)

3. Run the `main.py` file


4. press `q` to quit

 
## Additional files

   Since this is a course project, along with the actual implementation of the tool there're lots of theoritical work that I've attached in the `additional-files` folder in this repository. In these files PCA is thorougly explained as well as the methodology adopted from collecting the dataset to explaining the final results.
    
   Also, an image compression method based on PCA is presented, but the code isn't attached here since it's quiet messy.
    
   The `additional-files` folder includes:
    * the project report
    * 2-pages summary
    * scientific poster
    * the slides used in a video where I'm demonstrating what I've done in this project. [video link](https://nileuniversity-my.sharepoint.com/:v:/g/personal/g_gad_nu_edu_eg/Ef9wmxNNZ-hIpFQOoi-T9nMBkrgELqQzuk7HG-qnFL5fCw?e=FKGguw)
