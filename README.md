# Face-Emotion-Detection

**Project Introduction:**

The Indian education landscape has been undergoing rapid changes for the past ten years owing to the advancement of web-based learning services, specifically eLearning platforms.

Digital platforms might overpower physical classrooms in terms of content quality, but in a physical classroom, a teacher can see the faces and assess the emotion of the class and tune their lecture accordingly, whether he is going fast or slow. He can identify students who need special attention. While digital platforms have limitations in terms of physical surveillance, it comes with the power of data and machines, which can work for you.

It provides data in form of video, audio, and texts, which can be analyzed using deep learning algorithms. A deep learning-backed system not only solves the surveillance issue, but also removes the human bias from the system, and all information is no longer in the teacher’s brain but rather translated into numbers that can be analyzed and tracked.


**Objective**

Our objective is to solve the above mentioned challenge by applying deep learning algorithms to live video data inorder to recognize the facial emotions and categorize them accordingly.

**Dataset used:**

We have utilized the FER 2013 dataset provided on Kaggle.
The data consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that the face is more or less centred and occupies about the same amount of space in each image.

Dependencies:

Install these libraries before running the colab notebook.

1. numpy
2. streamlit==1.9.0
3. tensorflow-cpu==2.9.0
4. opencv-python-headless==4.5.5.64
5. streamlit-webrtc==0.37.0



Project Overview:


We start with downloading the required dataset from Kaggle. Once the data is available, the training and validation sets are defined.

The next step is to preprocess the datasets; this includes rescaling the data by multiplying it by 1/255 to obtain the target values in the range [0,1] and performing data augmentation for artificially creating new data. Data augmentation also helps to increase the size and introduce variability in the dataset.

After preparing the data, we construct the required CNN model using TensorFlow and Keras libraries to recognize the facial emotions of a user. This model consists of four convolutional layers and three fully connected layers to process the input image data and predict the required output. In between each layer, a Max Pooling and Dropout layer was added for downsampling the data and preventing our model from overfitting. Finally, for compiling all the layers, we have used the Adam optimizer, with loss function as Categorical Cross entropy and accuracy as the metric for evaluation.

Once the model was ready, we trained it using the prepared data.

The model achieved an accuracy of 77% on the training set and 64% on the validation set after fifteen epochs.


Finally, using an image passed through our model, we confirmed that it could correctly recognize the emotions.

Additionally, using OpenCV and Streamlit and Heroku, we created a web app to monitor live facial emotion recognition. The web app successfully identified each class and also was able to detect multiple faces and their respective emotions.

The link for the web app: https://real-time-face-emotion-detect.herokuapp.com/



**Demo video to display the working of our web app deployed on streamlit.**


[streamlit-main-2022-07-20-01-07-77.webm](https://user-images.githubusercontent.com/62935266/179839446-d5bbb4be-1169-4937-80ea-f99ca45a740f.webm)

