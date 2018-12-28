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


with open('results.txt', 'w') as file:
    for metric in metrics_list:
        file.write(metric)
        file.write(','.ljust(24))
    file.write('\n')

# Load the images, compute the confusion matrix and calculate the metrics
print('Loading the images...')
for (i, images) in enumerate(zip(segmentedPaths, gtPaths)):
    segmented = cv2.imread(images[0], 0)
    gt = cv2.imread(images[1], 0)

    segmented = cv2.resize(segmented, (gt.shape[1], gt.shape[0]))

    # Verify if the segmented and ground truth images have the same size
    if segmented.shape != gt.shape:
        print('The sizes of segmented image and ground truth image must be the same!')
        pass

    print('Image {}/{}'.format(i + 1, len(segmentedPaths)))

    # Threshold the images to be sure that the pixel values are binary
    ret, segmented = cv2.threshold(segmented, 127, 255, cv2.THRESH_BINARY)
    ret, gt = cv2.threshold(gt, 127, 255, cv2.THRESH_BINARY)

    # Get the confusion matrix of the image
    confusion_matrix = metrics.get_confusion_matrix(segmented, gt)

    values = []

    for metric in metrics_list:
        values.append(metrics.calculate_metric(metric, confusion_matrix))

    with open('results.txt', 'a') as f:
        for value in values:
            f.write('{:.5f}'.format(value))
            f.write(','.ljust(20))
        f.write('\n')
