import cv2
import argparse
import findpaths
import metrics
import csv

# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--segmented-images', type=str, required=True,
                help='Path to the segmented images.')
ap.add_argument('-g', '--ground-truth', type=str, required=True,
                help='Path to the ground truth images')
ap.add_argument('-r', '--results', type=str, required=False,
                help='Path to the outputs results. If not used the '
                     'results will be saved on the same path of the segmented images')
'''
 Verify if it is possible to print directly on LaTeX format

ap.add_argument('-o', '--output-type', type=str, required=False, default='csv',
                help='Type of the output file. Accepts txt, xls and csv.'
                     'If not used the results will be saved on csv format.')
'''
args = vars(ap.parse_args())

# Pick all the paths to the segmented images
try:
    segmentedPaths = list(findpaths.list_images(args['segmented_images']))
except Exception:
    print('Error on the path to the segmented images. Verify the paths carefully!')

# Pick all the paths to the ground truth images
try:
    gtPaths = list(findpaths.list_images(args['ground_truth']))
except Exception:
    print('Error on the path to the ground truth images. Verify the paths carefully!')

# Verify if the number of images are equal
if len(segmentedPaths) != len(gtPaths):
    print('The number of images to be compared must be of the same size!')
    exit()

metrics_list = ['Jac', 'Mcc', 'Dice', 'Acc', 'Sen', 'Spe', 'Ppv', 'Npv', 'Fpr', 'Fdr', 'Fnr', 'F1-Score']

# Create a dictionary that will contain all the metrics as keys and their respective values for each image
dict_values = {}

# Load the images, compute the confusion matrix and calculate the metrics
print('Loading the images...')
for (i, images) in enumerate(zip(segmentedPaths, gtPaths)):
    print('Image {}/{}'.format(i + 1, len(segmentedPaths)))

    segmented = cv2.imread(images[0], 0)
    gt = cv2.imread(images[1], 0)

    segmented = cv2.resize(segmented, (gt.shape[1], gt.shape[0]))

    # Verify if the segmented and ground truth images have the same size
    if segmented.shape != gt.shape:
        print('The sizes of segmented image and ground truth image must be the same!')
        pass

    # Threshold the images to be sure that the pixel values are binary
    ret, segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
    ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

    # Get the confusion matrix of the image
    confusion_matrix = metrics.get_confusion_matrix(segmented, gt)

    # Calculate all the metrics and put them into a dictionary
    dict_values = metrics.get_metrics(metrics_list, dict_values, confusion_matrix)

# Print the metrics for each image
with open('results.txt', 'w') as f:
    for metric in dict_values.keys():
        f.write(metric.ljust(10))
        for value in dict_values[metric]:
            f.write('{:.5f}'.format(value))
            f.write(','.ljust(10))
        f.write('\n')

# Print the mean of each metric for all images
with open('results.txt', 'a') as f:
    # Calculate the mean of each metric
    mean_values = metrics.mean(dict_values)
    f.write('\n\nMean Values: \n\n')
    for metric, mean in zip(dict_values.keys(), mean_values):
        f.write(metric.ljust(10))
        f.write('{:.5f}'.format(mean))
        f.write('\n')
