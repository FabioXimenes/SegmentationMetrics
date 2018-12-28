# SegmentationMetrics

This is a repository that contains code written in python for calculating the key metrics used to validate segmentation results.

## Prerequisites
  * Python 3
  * OpenCV 3.4.4
  * Numba 0.41.0
 
 PS.: Other versions of these libraries may work properly. 
 
## Metrics
 * Accuracy
 * Sensitivity
 * Specificity
 * Jaccard Index
 * Dice Coefficient
 * Matthew Correlation Coefficient
 * Precision
 * Negative Predictive Value
 * False Positive Rate
 * False Discovery Rate
 * False Negative Rate
 * F1-Score
 
 
 ### Usage:
 Separate your images into two folders. One folder must contain the images segmented by your method, and the other folder must contain
 the images segmented by a specialist. The order of the images in both folders must agree for the correct measure of the metrics.
 
 After that, clone this repository, and inside the folder, execute the code 'main.py' via command-line. You will have to inform the paths to the two folders recently created.
 
 Example of execution:
 $ python3 main.py -s segmentation -g specialist
 
 The first argument is the path to the folder where the segmented images are and the second argument is the path to folder containing the images segmented by the specialist. 
 
 ### Output
 The metrics will be printed in a 'results.txt' file.
 
