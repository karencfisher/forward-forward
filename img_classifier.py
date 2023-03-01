import time
import numpy as np
import tensorflow as tf

from layers import FFDense
import utils


class ImageClassifier:
    def __init__(self, extractor, layers, n_classes, learning_rate=0.03):
        self.extractor = extractor
        input_shape = self.extractor.output.shape[1]
        self.layer_list = []
        for i in range(len(layers)):
            shape = input_shape + n_classes if i == 0 else layers[i-1]
            self.layer_list += [
            FFDense(layers[i], 
                    activation='relu', 
                    input_shape=(shape,))
            ]
        
        self.learning_rate = learning_rate
        self.n_classes = n_classes

    def FF_fit(self, x, y):
        # Make x_pos by concatenating one-hot encoding of class with the latent 
        # image vectors
        y = y.numpy().tolist()
        one_hots = np.array(list(map(lambda x: utils.onehot_encode(x, self.n_classes), y)))
        x_pos = np.hstack([one_hots, x])

        # Make x_neg with shuffled labels
        random_y = tf.random.shuffle(y)
        one_hots = np.array(list(map(lambda x: utils.onehot_encode(x, self.n_classes), random_y)))
        x_neg = np.hstack([one_hots, x])
        
        # pass both through each FFDense layers
        loss_var = []
        for layer in self.layer_list:
            x_pos, x_neg, loss = layer.forward_forward(x_pos, x_neg)
            loss_var.append(loss.numpy())
        mean_res = sum(loss_var) / len(loss_var)
        return mean_res

    def FF_predict(self, x):
        # TBD
        good_per_label = []
        for y in range(self.n_classes):
            one_hot = utils.onehot_encode(y, self.n_classes)
            x_labeled = np.concatenate([one_hot, x])
            x_labeled = np.expand_dims(x_labeled, 0)
            h = x_labeled
            good_layers = []
            for layer in self.layer_list:
                h = layer(h)
                good = tf.math.reduce_mean(tf.math.pow(h, 2), 1).numpy()
                good_layers.append(good)
            good_per_label.append(np.mean(good_layers))
        return np.argmax(good_per_label)    

    def fit(self, x_batches, epochs=5):
        history = []
        num_examples = x_batches.reduce(0, lambda x, _: x + 1).numpy()
        gg = utils.GasGuage(num_examples)
        for epoch in range(epochs):
            print(f'Epoch {epoch + 1}')
            gg.begin()
            losses = []
            start_time = time.time()
            for count, batch in enumerate(x_batches):
                # Generate latent images
                image_batch, label_batch = batch
                latent_imgs = self.extractor.predict(image_batch, verbose=0)

                # Pass them through the FFDense layers
                mean_res = self.FF_fit(latent_imgs, label_batch)
                losses.append(mean_res)
                
                # update progress
                gg.update(count + 1)
            mean_losses = sum(losses) / len(losses)
            elapsed = time.time() - start_time
            gg.done(f'{utils.time_str(elapsed)} - loss: {mean_losses:.4f}')
            history.append(mean_losses)
        return {'loss': history}
    
    def predict(self, x_batches):
        labels = []
        num_examples = x_batches.reduce(0, lambda x, _: x + 1).numpy()
        gg = utils.GasGuage(num_examples)
        gg.begin()
        start_time = time.time()
        for count, batch in enumerate(x_batches):
            # Generate latent images
            image_batch, _ = batch
            latent_imgs = self.extractor.predict(image_batch, verbose=0)

            # Do inference through the FFDense layers
            for example in latent_imgs:
                label = self.FF_predict(example)
                labels.append(label)

            gg.update(count + 1)
        elapsed = time.time() - start_time
        gg.done(utils.time_str(elapsed))
        return labels
