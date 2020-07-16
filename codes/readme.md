Read Me
===========================

## Reference 
#### **Paper**  
Self-supervised Spatio-temporal Representation Learning for Videos
by Predicting Motion and Appearance Statistics. 

**Repo Link of the author** - https://github.com/ayushjain2699/video_repres_mas  

## Files 

### compute_motion_statistics_fast.py 
Compute the motion statistics of the video, to be used in the SSL pre-task. 

### input_data.py 
Program to input the data from a "list" file containing the list of input videos' directory structure of its RGB frames, u_flow and v_flow. 

### new_input_data.py 
Program to input the data using a "list" of videos for the SL approach. 

### model.py 
Contains the C3D model used for implementing the SSL approach. 

### model_class_det.py 
contains the model(two FC layers) used over the C3D network for the SL approach. 

### train.py 
Program for training the SSL approach. 

### train_fine_class_det.py 
Program for training the SL approach by using pre-initialized weightss from the SSL approach and then traianing the entire network. 

### train_full_class_det.py 
Program for training the SL approach by randomly initializing the weights of the C3D network. 

### train_class_det.py 
Program for training the SL approach using the pre-initialized weights from the SSL approach and then training only the two FC layers (freezing the C3D network).  

### testing_as_ftn.py
Program to test the models at number of iterations.  

### test.py
Program to test the model at a given iteration.  

### test_full_class_labels.py
Program to test the model with random initialization of weights.  

### test_class_labels.py
Program to test the model with pre-initialization of weights from SSL and freezing those layers.  

## Diretories  

### final  
Contains all the codes for the training and testing using the original crescent dataset.  
*Note* - Contains "Readme".  