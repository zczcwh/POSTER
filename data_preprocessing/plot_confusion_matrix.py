import itertools
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_confusion_matrix(cm, labels_name, title, acc):
    cm = cm / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, "{:0.2f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.imshow(cm, interpolation='nearest')
    plt.title(title)
    plt.colorbar()
    num_class = np.array(range(len(labels_name)))
    plt.xticks(num_class, labels_name, rotation=90)
    plt.yticks(num_class, labels_name)
    plt.ylabel('Target')
    plt.xlabel('Prediction')
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))
    plt.tight_layout()
    plt.savefig(os.path.join('./Confusion_matrix', title, "acc" + str(acc) + ".png"), format='png')
    plt.show()

