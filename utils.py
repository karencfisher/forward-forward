import random
import numpy as np
import matplotlib.pyplot as plt


def onehot_encode(label, n_classes):
    onehot = np.zeros(n_classes)
    onehot[label] = 1
    return onehot
    
def embed(x, y, neg=False):
    x_enc = np.copy(x)
    n_classes = np.max(y) + 1
    for i in range(x.shape[0]):
        label_enc = onehot_encode(y[i], n_classes)
        if neg:
            orig_enc = np.copy(label_enc)
            while np.all(orig_enc == label_enc):
                np.random.shuffle(label_enc)
        x_enc[i, 0, :n_classes] = label_enc[:]
    return x_enc

def show_samples(x, y, n_samps):
    imgs = []
    for _ in range(n_samps):
        idx = random.randint(0, x.shape[0] - 1)
        imgs.append((x[idx], y[idx]))
    
    plt.figure(figsize=(10, 10))
    for idx, item in enumerate(imgs):
        image, label = item
        plt.subplot(2, 2, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label : {label}")
    plt.show()