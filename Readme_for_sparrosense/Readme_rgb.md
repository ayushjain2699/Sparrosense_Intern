---
output:
  pdf_document: default
  html_document: default
---
Read me  
=====================

## Files  

### Unique_videos.list  
Contains the list of all the videos from the crescent dataset.  
*Format* - \<Directory structure of the video>  

### list_SSL_home_dir.list  
Contains the list of 256 selected vides to be used for the SSL training.  
*Format* - \<Directory structure of the extracted RGB frames of the video>  

### make_frames.py  
Program to extract the RGB Frames of the videos.  

### calc_optical_flow.py  
Program to extract the u_flow and v_flow images of the videos.  

## Directories  

Contains the extracted RGB frames and optical Flow images of the videos, organized date-wise.  