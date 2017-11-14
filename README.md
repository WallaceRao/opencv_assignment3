# opencv_assignment3
Gesture recognition for camera and image


/*
 *                                SPECIFICATION
 *  The programe aims to recognise gesture in an image. You can specify a static image
 *  for recognising, or capture image via camera without specifing any parameter.
 *  The processing workflow is
 *  #1 If there is a saved MLP 'bp.xml' under the same dir, load it
 *  #2 If there is no 'bp.xml' under the same dir, train the MLP
 *  #3 Preprocess the static image or frame from camera by following steps:
 *      1) convert the image to gray
 *      2) threshold, filter all pixels whose grey scale is larger than 230 or small than 70 to
 *         avoid the affection by background
 *      3) Median filter to avoid affection by noise
 *      4) Threshold with specifing CV_THRESH_OTSU to extract the gesture
 *      5) Find the largest contour
 *      6) Calculate the CEs from the contour
 *      7) Predict the result via the MLP
 *  #4 Show result 
 * 
 * 
 *                                   NOTE
 *  #1 To train the MLP, there should be a folder named "trainImgs" under the same directory
 *     with the executable. All train images should putted be in the folder "trainImgs".
 *  #2 All names of all train images should contain at least 7 characters, and the 7th
 *     char should be the gesture number, for example, the train image
 *     "hand1_0_bot_seg_1_cropped.png" means the gesture of this image representes '0'.
 *  #3 The train process may take a few seconds to complete.
 *  #4 To recognise gesture via camera, you should make sure the background of the hand
 *     is black(or the grey scale is very different with the hand), so that the gesture
 *     could be extracted correctly.
 */
 
 Any question, contact raoyonghui0630@gmail.com
