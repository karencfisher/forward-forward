import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd


class Normalize:
    def fit(self, x):
        self.avg = np.mean(x)
        self.std = np.std(x)

    def transform(self, x):
        return (x - self.avg) / self.std
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    
class GasGuage:
    def __init__(self, n_steps, percentage=5):
        self.n_steps = n_steps
        self.percentage = percentage
        self.size = 100 // percentage
        
    def begin(self):
        print(f'0/{self.n_steps}[{" " * self.size}]', end='')

    def update(self, step):
        percent = int(step / self.n_steps * 100)
        done = percent // self.percentage
        left = self.size - done
        print(f'\r{step}/{self.n_steps}[{"=" * done}{" " * left}]', end='')

    def done(self, text):
        print(f' {text}')   

    
def time_str(seconds):
        hr = int(seconds / 3600)
        seconds %= 3600
        min = int(seconds / 60)
        sec = seconds % 60
        if hr > 0:
            output = f'{hr} hr {min} min {sec:.2f} sec'
        elif min > 0:
            output = f'{min} min {sec:.2f} sec'
        else:
            output = f'{sec:.4f} sec'
        return output

def show_samples(x, y, labels=None):
    imgs = []
    for _ in range(4):
        idx = random.randint(0, x.shape[0] - 1)
        imgs.append((x[idx], y[idx]))
    
    plt.figure(figsize=(7, 7))
    for idx, item in enumerate(imgs):
        image, label = item
        if labels is not None:
            label = labels[label]
        plt.subplot(2, 2, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Label : {label}")
        plt.xticks([])
        plt.yticks([])
    plt.show()
    
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    axes[0].plot(history['loss'], color='red')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')

    accuracy = history.get('accuracy')
    if accuracy is not None:
        axes[1].plot(accuracy, color='green')
        axes[1].set_title('Training Accuracy')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')

    plt.show()

def evaluate(y_true, y_pred, class_names=None, figsize=20, desc=''):
    accuracy = accuracy_score(y_true, y_pred)
    print(f'{desc} Accuracy = {accuracy * 100:.2f}%')
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(figsize, figsize))
    if class_names is None:
        class_names = [str(i) for i in range(max(y_true) + 1)]
    sns.heatmap(cm,
                annot=True,
                fmt='.2f',
                cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title(f'{desc} Confusion Matrix')
    plt.show()

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

def compare(predicted, actual, data, misses_df, classes=None):
    x_test, y_hat, y_test = data
    cond = (misses_df['Predicted'] == predicted) & (misses_df['Actual'] == actual)
    examples = misses_df[cond]['Example']

    samps = random.sample(list(examples), 4)
    imgs = []
    for samp in samps:
        imgs.append((x_test[samp], y_hat[samp], y_test[samp]))

    plt.figure(figsize=(10, 10))
    for idx, item in enumerate(imgs):
        image, predlabel, label = item
        if classes is not None:
            predlabel, label = classes[predlabel], classes[label]
        plt.subplot(2, 2, idx + 1)
        plt.imshow(image, cmap="gray")
        plt.title(f"Predicted: {predlabel} Actual: {label}")
    plt.show()

def get_worst(y_hat, y_test, num_errors):
    misses_df = pd.DataFrame(columns=['Predicted', 'Actual'])
    misses_df['Predicted'] = y_hat
    misses_df['Actual'] = y_test
    cond = misses_df['Predicted'] != misses_df['Actual']
    misses_df = misses_df[cond]
    misses_df = misses_df.sort_values(by=['Predicted', 'Actual'])
    misses_df.reset_index(inplace=True)
    misses_df = misses_df.rename(columns = {'index':'Example'})
    top_errors = misses_df.groupby(['Predicted', 'Actual']).size().sort_values(ascending=False).head(num_errors)
    return misses_df, top_errors
