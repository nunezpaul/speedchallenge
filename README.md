# speedchallenge

## Performance

I'll be keeping track of the performance of my model over time. Ideally the ideas I'll implement will improve the 
performance on the validation set. 

- [2019 01 07] Currently the model performs well on image pairs taken from the same video but not 
so well on the validation video I found. I suspect that the black bars on the sides of the image confuse the model.  
This would make sense since the black bars are constant and it will attempt to classify the speed based on the 
differences between two image frames. I can prove this if I crop out the right edge and retrain the model on the 
cropped images. 

[Update] Cropping out the black bars out from the edge of the image does not improve performance. Most 
likely that the difference brightness levels are causing the difference. Will look into further brightness and gamma 
regularization. 

[Update2] Just noticed that the rate at which the videos play back are not consistent. Looking the speed 
at which the person crosses the crosswalk at the beginning of the validation video, it's clear that the video is sped
up. With the training of the model it tends to equilibrate to a categorical accuracy of 20%. Inspecting the data, we 
find that there's nearly ~18% labeled at the minimum speed and ~2% at the maximum speed. I suspect that most of the 
correct labels come from correctly identifying these two classes. Will test by producing a validation dataset that is 
50% comprised of the min and max speed classes. If my hypothesis is correct 


| Date | Training Performance (MSE / Accuracy) | Validation Performance (MSE / Accuracy) |
|:---:|:---:|:---:|
2019 01 07| 2.2 / 88% | 90 / 25 % |

* Note that the performance is taken as the best for their respective sets of data

## References
- https://www.cs.ox.ac.uk/files/9026/DeepVO.pdf
- https://github.com/ChiWeiHsiao/DeepVO-pytorch
- https://github.com/experiencor/speed-prediction/blob/master/Dashcam%20Speed%20-%20C3D.ipynb
- http://cs229.stanford.edu/proj2017/final-reports/5244226.pdf
- https://github.com/bpenchas/video2speed
- https://github.com/JonathanCMitchell/speedChallenge