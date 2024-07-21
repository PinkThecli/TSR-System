import tensorflow as tf


class Localization(tf.keras.layers.Layer):
    def __init__(self, conv1_maps=20, conv1_shape=(5, 5), conv2_maps=20, conv2_shape=(5, 5), dense_units=20):
        super(Localization, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(conv1_maps, conv1_shape, strides=1, padding="same", activation="relu")
        self.pool1 = tf.keras.layers.MaxPool2D((2, 2), strides=2)
        self.conv2 = tf.keras.layers.Conv2D(conv2_maps, conv2_shape, strides=1, padding="same", activation="relu")
        self.pool2 = tf.keras.layers.MaxPool2D((2, 2), strides=2)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(dense_units, activation="relu")
        self.fc2 = tf.keras.layers.Dense(6, activation=None, bias_initializer=tf.keras.initializers.constant([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), kernel_initializer="zeros")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        theta = self.fc2(x)
        theta = tf.keras.layers.Reshape((2, 3))(theta)
        return theta


class BilinearInterpolation(tf.keras.layers.Layer):
    def __init__(self, height, width):
        super(BilinearInterpolation, self).__init__()
        self.height = height
        self.width = width

    def call(self, inputs):
        images, theta = inputs
        homogenous_coordinates = self.grid_generator(batch=tf.shape(images)[0])
        return self.interpolate(images, homogenous_coordinates, theta)

    def grid_generator(self, batch):
        x = tf.linspace(-1, 1, self.width)
        y = tf.linspace(-1, 1, self.height)
        xx, yy = tf.meshgrid(x, y)
        xx = tf.reshape(xx, (-1,))
        yy = tf.reshape(yy, (-1,))
        homogenous_coordinates = tf.stack([xx, yy, tf.ones_like(xx)])
        homogenous_coordinates = tf.expand_dims(homogenous_coordinates, axis=0)
        homogenous_coordinates = tf.tile(homogenous_coordinates, [batch, 1, 1])
        homogenous_coordinates = tf.cast(homogenous_coordinates, dtype=tf.float32)
        return homogenous_coordinates

    def interpolate(self, images, homogenous_coordinates, theta):
        transformed = self.batch_matmul(theta, homogenous_coordinates)
        transformed = tf.transpose(transformed, perm=[0, 2, 1])
        transformed = tf.reshape(transformed, [-1, self.height, self.width, 2])

        x_transformed = transformed[:, :, :, 0]
        y_transformed = transformed[:, :, :, 1]

        x = ((x_transformed + 1.0) * tf.cast(self.width, dtype=tf.float32)) * 0.5
        y = ((y_transformed + 1.0) * tf.cast(self.height, dtype=tf.float32)) * 0.5

        x0 = tf.cast(tf.math.floor(x), dtype=tf.int32)
        x1 = x0 + 1
        y0 = tf.cast(tf.math.floor(y), dtype=tf.int32)
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, 0, self.width - 1)
        x1 = tf.clip_by_value(x1, 0, self.width - 1)
        y0 = tf.clip_by_value(y0, 0, self.height - 1)
        y1 = tf.clip_by_value(y1, 0, self.height - 1)
        x = tf.clip_by_value(x, 0, tf.cast(self.width, dtype=tf.float32) - 1.0)
        y = tf.clip_by_value(y, 0, tf.cast(self.height, dtype=tf.float32) - 1.0)

        i_a = self.advance_indexing(images, x0, y0)
        i_b = self.advance_indexing(images, x0, y1)
        i_c = self.advance_indexing(images, x1, y0)
        i_d = self.advance_indexing(images, x1, y1)

        x0 = tf.cast(x0, dtype=tf.float32)
        x1 = tf.cast(x1, dtype=tf.float32)
        y0 = tf.cast(y0, dtype=tf.float32)
        y1 = tf.cast(y1, dtype=tf.float32)

        w_a = (x1-x) * (y1-y)
        w_b = (x1-x) * (y-y0)
        w_c = (x-x0) * (y1-y)
        w_d = (x-x0) * (y-y0)

        w_a = tf.expand_dims(w_a, axis=3)
        w_b = tf.expand_dims(w_b, axis=3)
        w_c = tf.expand_dims(w_c, axis=3)
        w_d = tf.expand_dims(w_d, axis=3)

        return tf.math.add_n([w_a*i_a + w_b*i_b + w_c*i_c + w_d*i_d])

    def batch_matmul(self, a, b):
        a_row, a_col = a.shape[1], a.shape[2]
        b_row, b_col = b.shape[1], b.shape[2]
        tiled_a = tf.reshape(tf.tile(a, [1, b_col, 1]), shape=[-1, b_col, a_row, a_col])
        tiled_b = tf.reshape(tf.tile(b, [1, 1, a_row]), shape=[-1, b_row, a_row, b_col])
        return tf.reduce_sum(tf.transpose(tiled_a, [0, 2, 3, 1]) * tf.transpose(tiled_b, [0, 2, 1, 3]), axis=2)

    def advance_indexing(self, inputs, x, y):
        shape = tf.shape(inputs)
        batch_size, _, _ = shape[0], shape[1], shape[2]
        batch_idx = tf.range(0, batch_size)
        batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
        b = tf.tile(batch_idx, (1, self.height, self.width))
        indices = tf.stack([b, y, x], 3)
        return tf.gather_nd(inputs, indices)


class SpatialTransform(tf.keras.layers.Layer):
    def __init__(self, conv1_maps=20, conv1_shape=(5, 5), conv2_maps=20, conv2_shape=(5, 5), dense_units=20, **kwargs):
        super(SpatialTransform, self).__init__(**kwargs)
        self.conv1_maps = conv1_maps
        self.conv1_shape = conv1_shape
        self.conv2_maps = conv2_maps
        self.conv2_shape = conv2_shape
        self.dense_units = dense_units

    def build(self, input_shape):
        self.localization = Localization(self.conv1_maps, self.conv1_shape, self.conv2_maps, self.conv2_shape, self.dense_units)
        self.bilinear_intepolation = BilinearInterpolation(input_shape[1], input_shape[2])

    def call(self, inputs):
        theta = self.localization(inputs)
        return self.bilinear_intepolation([inputs, theta])
