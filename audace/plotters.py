import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
import numpy as np
import itertools
from pathlib import Path
import math


def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    dir_path = Path('./figures')

    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    path = Path(dir_path, fig_id + "." + fig_extension)

    if tight_layout:
        plt.tight_layout()

    plt.savefig(path, format=fig_extension, dpi=resolution)


def _brightness(colormap, value):
    r, g, b, t = colormap(value)
    return math.sqrt(.241*r*r + .691*g*g + .068*b*b)


def plot_confusion_matrix2(cm,
                           target_names,
                           title='Confusion matrix',
                           cmap=None,
                           normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed
                  from matplotlib.pyplot.cm
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,
                          normalize    = True,
                          target_names = y_labels_vals,
                          title        = best_estimator_name)

    Reference
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    http://matplotlib.org/examples/color/colormaps_reference.html

    """

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    # plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    # plt.show()


def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    if normalize:
        cm = 100.0*cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        text_color = "white" if cm[i, j] > thresh else "black"
        if normalize:
            plt.text(j, i, "{:0.2f}%".format(cm[i, j]),
                     horizontalalignment="center",
                     color=text_color)
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]).replace(',', ' '),
                     horizontalalignment="center",
                     color=text_color)

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(
        accuracy, misclass))
    # plt.show()


def plot_roc_curve(clf, X, y, label=None):
    margin = 0.01
    y_scores = clf.decision_function(X)
    fpr, tpr, thresholds = roc_curve(y, y_scores)
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')  # dashed diagonal
    plt.axis([0-margin, 1+margin, 0-margin, 1+margin])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=14)
    plt.ylabel('True Positive Rate (Recall)', fontsize=14)
    plt.grid(True)


def plot_precision_recall_vs_threshold(clf, X, y):
    vmargin = 0.01
    y_scores = clf.decision_function(X)

    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

    plt.plot(thresholds,
             precisions[:-1],
             "r--",
             label="Precision",
             linewidth=2
             )

    plt.plot(thresholds,
             recalls[:-1],
             "g--",
             label="Recall",
             linewidth=2
             )

    plt.legend(loc="center right", fontsize=14)
    plt.xlabel("Threshold", fontsize=14)
    plt.grid(True)
    plt.axis([thresholds.min(), thresholds.max(), -vmargin, 1+vmargin])


def plot_precision_vs_recall(clf, X, y):
    margin = 0.01

    y_scores = clf.decision_function(X)
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.axis([0-margin, 1+margin, 0-margin, 1+margin])
    plt.grid(True)


def clf_full_report(clf, X, y, target_names, save_as=None):
    y_pred = clf.predict(X)
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y, y_pred)

    plt.figure(figsize=(12, 15))

    plt.subplot(321)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix,
                          target_names=target_names,
                          normalize=False,
                          title='Confusion matrix, without normalization')

    plt.subplot(322)
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix,
                          target_names=target_names,
                          normalize=True,
                          title='Normalized confusion matrix')

    plt.subplot(323)
    # Plot Precision vs Recall
    plot_precision_vs_recall(clf, X, y)

    plt.subplot(324)
    # Plot Receiver Operating Characteristic
    plot_roc_curve(clf, X, y)

    plt.subplot(313)
    # Plot precision and recall vs threshold over 2 columns
    plot_precision_recall_vs_threshold(clf, X, y)

    plt.tight_layout()

    if save_as:
        save_fig(save_as)

    plt.show()
