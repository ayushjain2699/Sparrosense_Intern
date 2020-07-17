---
output:
  pdf_document: default
  html_document: default
---
Read me
===================

## Files  

### list_for_SSL.list
Contains the list of videos(256 videos) used for the SSL.  

### list_SL.list
Contains the list of videos(214 examples for each of the five classes) used for the SL.  
*Format* - \<directory structure of start video>\<space>\<directory structure of end video>\<class_number>  
\<space>\<start_frame>\<space>\<end_frame>.   

### latest_SL_train_new.list
Contains the list of videos(500 examples for each of the five classes) used for the SL.  
Note that for the class "spray coat", Since it has only 214 examples, 500 examples of this class contains duplicate values.  
*Format* - \<directory structure of start video>\<space>\<directory structure of end video>\<class_number>  
\<space>\<start_frame>\<space>\<end_frame>.  

### test_SL_new.list  
*Against training list -  list_SL.list*  
Contains the list of test examples(2302 examples).  
*Format* - \<directory structure of start video>\<space>\<directory structure of end video>\<class_number>  
\<space>\<start_frame>\<space>\<end_frame>.  

### test.list
*Against trainin list - latest_SL_new.list*  
Contains the list of test examples(693 examples)  
*Format* - \<directory structure of start video>\<space>\<directory structure of end video>\<class_number>  
\<space>\<start_frame>\<space>\<end_frame>.  
