import tensorflow as tf
import matplotlib.pyplot as plt
import pickle

def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')

    return fig, axes

with open('train_samples.pkl', 'rb') as f:
    samples = pickle.load(f)

_ = view_samples(-1, samples)
plt.show()