import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    confusion_matrix
)

import numpy as np
import itertools
from pathlib import Path
import math


def save_fig(exp_name, fig_id, tight_layout=True, ext="png", res=300):
    dir_path = Path(exp_name, 'figures')

    if not dir_path.exists():
        dir_path.mkdir(parents=True)

    path = Path(dir_path, fig_id + "." + ext)

    if tight_layout:
        plt.tight_layout()

    plt.savefig(path, format=ext, dpi=res)


def _brightness(colormap, value):
    r, g, b, t = colormap(value)
    return math.sqrt(.241*r*r + .691*g*g + .068*b*b)


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


def plot_x_vs_y(x_title, x, y_title, y):
    plt.title(x_title + " vs " + y_title)
    plt.plot(x, x, "k--", linewidth=0.8)
    plt.plot(x, y, "r", linewidth=2)
    plt.xlabel(x_title, fontsize=14)
    plt.ylabel(y_title, fontsize=14)
    plt.grid(True)


def plot_metric_vs_metric(h, x_metric, y_metric):
    x = h.history[x_metric]
    y = h.history[y_metric]
    plt.title("model " + x_metric + " vs " + y_metric)
    plt.plot(x, y, "r", linewidth=2)
    plt.xlabel(x_metric, fontsize=14)
    plt.ylabel(y_metric, fontsize=14)
    plt.grid(True)


def plot_precision_vs_recall(clf, X, y):
    margin = 0.01

    y_scores = clf.decision_function(X)
    precisions, recalls, thresholds = precision_recall_curve(y, y_scores)

    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.axis([0-margin, 1+margin, 0-margin, 1+margin])
    plt.grid(True)


def clf_full_report(clf, X, y, target_names, exp_name=None, save_as=None):
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
        save_fig(exp_name, save_as)

    plt.show()


def nn_full_report(m, param_X, param_y, target_names, exp_name=None, save_as=None):
    y_pred = m.predict(param_X).round()
    # Compute confusion matrix
    if param_y.shape[1] > 1:
        y = param_y.argmax(axis=1)
        y_pred = y_pred.argmax(axis=1)
    else:
        y = param_y

    cnf_matrix = confusion_matrix(y, y_pred)

    plt.figure(figsize=(12, 5))

    plt.subplot(121)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(cnf_matrix,
                          target_names=target_names,
                          normalize=False,
                          title='Confusion matrix, without normalization')

    plt.subplot(122)
    # Plot normalized confusion matrix
    plot_confusion_matrix(cnf_matrix,
                          target_names=target_names,
                          normalize=True,
                          title='Normalized confusion matrix')

    plt.tight_layout()

    if save_as:
        save_fig(exp_name, save_as)

    plt.show()


def plot_accuracy_curve(h):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='lower right')


def plot_loss_curve(h):
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['train', 'val'], loc='upper right')


def plot_accuracy_loss_curve(h):
    plt.plot(h.history['accuracy'])
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_accuracy'])
    plt.plot(h.history['val_loss'])
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(['accuracy', 'loss', 'val_acc', 'val_loss'], loc='center right')
    plt.gca().set_ylim(0, 1)
    plt.tight_layout()


def plot_nn_metrics(h):
    plt.title('model metrics')
    keys = h.history.keys()
    for key in keys:
        plt.plot(h.history[key])

    plt.ylabel('metric')
    plt.xlabel('epoch')
    plt.grid()
    plt.legend(keys, loc='center right')


def plot_nn_learning_curves(history, exp_name=None, save_as=None):
    plt.figure(figsize=(14, 8))

    plt.subplot(221)
    plot_accuracy_curve(history)

    plt.subplot(222)
    plot_loss_curve(history)

    plt.subplot(223)
    plot_metric_vs_metric(history, 'accuracy', 'loss')

    plt.subplot(224)
    plot_metric_vs_metric(history, 'val_accuracy', 'val_loss')

    plt.tight_layout()

    if save_as:
        save_fig(exp_name, save_as)

    plt.show()
