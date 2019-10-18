from numba import njit
from functools import reduce

# The numba package speeds up the access of the pixel inside the for loops.


@njit
def get_confusion_matrix(segmented, ground_truth):
    """

    :param segmented: The image segmented by your method
    :param ground_truth: The image segmented by a specialist
    :return: A dictionary with all the values from the confusion matrix
    (TP, TN, FP, FN).
    """

    # TODO - normalize the images to have pixel values between 0 and 1

    # Initialize the values of confusion matrix
    tp = tn = fp = fn = 0

    # Compute the values from confusion matrix
    for col in range(segmented.shape[0]):
        for row in range(segmented.shape[1]):
            if segmented[col, row] == 255 and ground_truth[col, row] == 255:
                tp += 1
            elif segmented[col, row] == 0 and ground_truth[col, row] == 0:
                tn += 1
            elif segmented[col, row] == 255 and ground_truth[col, row] == 0:
                fp += 1
            elif segmented[col, row] == 0 and ground_truth[col, row] == 255:
                fn += 1

    return tp, tn, fp, fn


def get_metrics(metrics_lst, dict_values, confusion_matrix):

    if not dict_values:
        for metric in metrics_lst:
            dict_values[metric] = list()

    for metric in metrics_lst:
        if metric in dict_values:
            dict_values[metric].append(calculate_metric(metric, confusion_matrix))

    return dict_values


def calculate_metric(name, confusion_matrix):
    """"

    :param name: metric the will be calculated
    :param confusion_matrix: tuple with the values of the confusion matrix (tp, tn, fp, fn)
    :return: the value of the metric
    """

    # TODO - verify if the name is valid. And raise an error if do not.

    if name == 'Jac':
        return jaccard_index(confusion_matrix)
    elif name == 'Mcc':
        return mcc(confusion_matrix)
    elif name == 'Dice':
        return dice_coefficient(confusion_matrix)
    elif name == 'Sen':
        return sensitivity(confusion_matrix)
    elif name == 'Spe':
        return specificity(confusion_matrix)
    elif name == 'Acc':
        return accuracy(confusion_matrix)
    elif name == 'Ppv':
        return ppv(confusion_matrix)
    elif name == 'Npv':
        return npv(confusion_matrix)
    elif name == 'Fpr':
        return fpr(confusion_matrix)
    elif name == 'Fdr':
        return fdr(confusion_matrix)
    elif name == 'Fnr':
        return fnr(confusion_matrix)
    elif name == 'F1-Score':
        return f1_score(confusion_matrix)


def mean(data):

    for metric in data.keys():
        metric_values = data[metric]
        value = reduce(lambda x, y: x + y, metric_values)/len(metric_values)

        yield value


def jaccard_index(confusion_matrix):
    """
    Return the value of Jaccard Index, a similarity metric.
    The value returned is between 0 and 1.

    :param confusion_matrix:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix

    return tp/(fp + fn + tp)


def mcc(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix

    return (tp * tn - fp * fn)/(((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)


def dice_coefficient(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix

    return 2*tp/(2*tp + fp + fn)


def accuracy(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix

    return (tp + tn)/(tp + tn + fp + fn)


def sensitivity(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix

    return tp/(tp + fn)

def specificity(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """
    tp, tn, fp, fn = confusion_matrix

    return tn/(tn + fp)


def f1_score(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """

    tp, tn, fp, fn = confusion_matrix

    return 2 * (tp/(tp + fp))*(tp/(tp + fn))/(tp/(tp + fp) + tp/(tp + fn))


def ppv(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """

    tp, tn, fp, fn = confusion_matrix

    return tp/(tp + fp)


def npv(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """

    tp, tn, fp, fn = confusion_matrix

    return tn/(tn + fn)


def fpr(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """

    tp, tn, fp, fn = confusion_matrix

    return fp/(fp + tn)


def fdr(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """

    tp, tn, fp, fn = confusion_matrix

    return fp/(fp + tp)


def fnr(confusion_matrix):
    """

    :param confusion_matrix:
    :return:
    """

    tp, tn, fp, fn = confusion_matrix

    return fn/(fn + tp)
