# speedchallenge

## Performance

I'll be keeping track of the performance of my model over time. Ideally the ideas I'll implement will improve the 
performance on the validation set. 

- [2019 01 07] Currently the model performs well on image pairs taken from the same video but not 
so well on the validation video I found. I suspect that the black bars on the sides of the image confuse the model.  
This would make sense since the black bars are constant and it will attempt to classify the speed based on the 
differences between two image frames. I can prove this if I crop out the right edge and retrain the model on the 
cropped images.

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