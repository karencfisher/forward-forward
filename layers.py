import numpy as np
import tensorflow as tf
from tensorflow import keras


class FFDense(keras.layers.Layer):
    def __init__(self, units, activation, input_shape=None, lr=0.03, thresh=1.5, 
                 iterations=50, **kwargs):
        super().__init__(**kwargs)
        self.thresh = thresh
        self.iterations = iterations
        self.loss_metric = keras.metrics.Mean()
        self.optimizer = keras.optimizers.Adam(learning_rate=lr)

        if input_shape is None:
            self.dense = keras.layers.Dense(
                units=units,
                activation=activation
            )
        else:
            self.dense = keras.layers.Dense(
                units=units,
                activation=activation,
                input_shape=input_shape
            )


    def call(self, x):
        x_norm = tf.norm(x, ord=2, axis=1, keepdims=True)
        x_norm += 1e-4
        x_dir = x / x_norm
        res = self.dense(x_dir)
        return res

    def forward_forward(self, x_pos, x_neg):
        for _ in range(self.iterations):
            with tf.GradientTape() as tape:
                g_pos = tf.math.reduce_mean(tf.math.pow(self.call(x_pos), 2), 1)
                g_neg = tf.math.reduce_mean(tf.math.pow(self.call(x_neg), 2), 1)
                loss = tf.math.log( 1 + tf.math.exp(
                    tf.concat([-g_pos + self.thresh, g_neg - self.thresh], 0)
                ))
                mean_loss = tf.cast(tf.math.reduce_mean(loss), tf.float32)
                self.loss_metric.update_state([mean_loss])
            gradients = tape.gradient(mean_loss, self.dense.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.dense.trainable_weights))
        return (
            tf.stop_gradient(self.call(x_pos)),
            tf.stop_gradient(self.call(x_neg)),
            self.loss_metric.result()
        )
