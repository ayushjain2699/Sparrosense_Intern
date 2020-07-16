---
output:
  pdf_document: default
  html_document: default
---
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

### model_new.py 
Contains the C3D model used for implementing the SSL approach. 

### model_class_det.py 
contains the model(two FC layers) used over the C3D network for the SL approach. 

### train_new.py 
Program for training the SSL approach. 

### train_fine_class_det.py 
Program for training the SL approach by using pre-initialized weightss from the SSL approach and then traianing the entire network. 

### train_full_class_det.py 
Program for training the SL approach by randomly initializing the weights of the C3D network. 

### train_class_det.py 
Program for training the SL approach using the pre-initialized weights from the SSL approach and then training only the two FC layers (freezing the C3D network). 

## Directories  

### motion_pattern_all_new_global 
Directory containing the saved checkpoints of the SSL model. 

### class_pred_model 
Directory containing the saved checkpoints of the SL model when only the FC layers are trained i.e. the C3D network is freezed with the preinitialized weights from the SSL approach. 

### temp_class_pred_model 
Directory containing the saved checkpoints of the SL model when only the FC layers are trained i.e. the C3D network is freezed with the preinitialized weights from the SSL approach, But for higher number of iterations. 

### fine_class_pred_model 
Directory containing the saved checkpoints of the SL model using the pre-initialized weights. 
Also contains the test.py function for testing the accuracy at various iterations. 

### full_class_pred_model 
Directory containing the saved checkpoints of the SL model using randomly initialized weights. 
Also contains the test.py function for testing the accuracy at various iterations. 

### datasets 
Contains the dataset(RGB frames and the optical flow extracted images) of the UCF101 dataset to be used for the SSL and SL approach. 

### list
Contains the ".list" files to be used for the input for training and testing the SSL and SL.  
*Note* - Contains "Readme".  

### rgb_orig_videos  
Contains the RGB and optical flow images of the original cresent data set.  
*Note* - Contains "Readme"  

### final  
Contains all the codes for the training and testing using the original crescent dataset.  
*Note* - Contains "Readme".  