Read me
==================

## Files 
 
### comput_motion_statistics_fast.py  
Computer the motion statistics of the video, to be used in the SSL pre-task.  

### input_SL_orig.py  
Program to input the data from the "list" of videos for the SL approach.  
*Returns* - the training clips(16 consecutive frames), target_label(class number) and next_batch_start.  

### input_SSL_orig.py  
Program to input the data from the "list" of videos for the SSL approach.  
*Returns* - the training clips(16 consecutive frames), target_label(motion statistics) and next_batch_start.  

### model_class_det.py 
contains the model(two FC layers) used over the C3D network for the SL approach.  

### model.py 
Contains the C3D model used for implementing the SSL approach. 

### test_SL.py
Program to test the SL model at by loading a specific checkpoint.  
**You can load a specific saved model and checkpoint by changing at line number 10 and 11.**    

### testing_ftn_full_SL.py
Program to test the SL model trained from scratch(random initialization) at a number of iterations.  
**You can load a specific saved model and checkpoint by changing at line number 10 and 29.**  

### testing_ftn_SL.py
Program to test the SL model trained using pre-initialized weights at a number of iterations.  
**You can load a specific saved model and checkpoint by changing at line number 11 and 30.**  

### train_full_SL.py
Program for training the SL model from scratch. 
**Note: Change the input video list and save path accordingly**

### train_SL.py
Program for training the SL model using pre-initialized weights.  
**Note: Change the input video list and save path accordingly**

### train_SSL.py
Program for training the SSL model.  
**Note: Change the input video list and save path accordingly**  