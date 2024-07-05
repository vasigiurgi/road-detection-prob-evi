import tensorflow as tf

class DS1(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(DS1, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.w = self.add_weight(
            name='Prototypes',
            shape=(self.units, input_shape[-1]),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        for i in range(self.units):
            if i == 0:
                un_mass_i = tf.subtract(self.w[i, :], inputs, name=None)
                un_mass_i = tf.square(un_mass_i, name=None)
                un_mass_i = tf.reduce_sum(un_mass_i, axis=-1)
                un_mass = tf.expand_dims(un_mass_i, -1)
            if i >= 1:
                un_mass_i = tf.subtract(self.w[i, :], inputs, name=None)
                un_mass_i = tf.square(un_mass_i, name=None)
                un_mass_i = tf.expand_dims(tf.reduce_sum(un_mass_i, axis=-1), -1)
                un_mass = tf.concat([un_mass, un_mass_i], -1)
        return un_mass

    def get_config(self):
        config = super(DS1, self).get_config()
        config.update({'units': self.units})
        return config


class DS1_activate(tf.keras.layers.Layer):
    def __init__(self, rate=1e-2, **kwargs):
        super(DS1_activate, self).__init__(**kwargs)
        self.rate = rate

    def build(self, input_shape):
        self.xi = self.add_weight(
            name='xi',
            shape=(1, input_shape[-1]),
            initializer='random_normal',
            trainable=True
        )

        self.eta = self.add_weight(
            name='eta',
            shape=(1, input_shape[-1]),
            initializer='random_normal',
            trainable=True
        )

    def call(self, inputs):
        gamma = tf.square(self.eta, name=None)
        gamma = tf.add(gamma, 1, name=None)
        alpha = tf.negative(self.xi, name=None)
        alpha = tf.exp(alpha, name=None) + 1
        alpha = tf.divide(1, alpha, name=None)
        si = tf.multiply(gamma, inputs, name=None)
        si = tf.negative(si, name=None)
        si = tf.exp(si, name=None)
        si = tf.multiply(si, alpha, name=None)
        self.add_loss(self.rate * tf.reduce_sum(alpha))
        return si

    def get_config(self):
        config = super(DS1_activate, self).get_config()
        config.update({'rate': self.rate})
        return config


class DS2(tf.keras.layers.Layer):
    def __init__(self, num_class, **kwargs):
        super(DS2, self).__init__(**kwargs)
        self.num_class = num_class

    def build(self, input_shape):
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[-1], self.num_class),
            initializer='random_normal',
            trainable=True
        )
        self.input_dim = input_shape[-1]

    def call(self, inputs):
        beta = tf.square(self.beta, name=None)
        beta_sum = tf.reduce_sum(beta, 1, keepdims=True)
        u = tf.divide(beta, beta_sum, name=None)
        inputs_new = tf.expand_dims(inputs, -1)
        a = inputs_new
        for j in range(self.num_class - 1):
            a = tf.concat([a, inputs_new], -1)
        inputs_new = a
        for i in range(self.input_dim):
            if i == 0:
                mass_prototype_i = tf.multiply(u[i, :], inputs_new[:, :, :, i, :], name=None)
                mass_prototype = tf.expand_dims(mass_prototype_i, -2)
            if i > 0:
                mass_prototype_i = tf.multiply(u[i, :], inputs_new[:, :, :, i, :], name=None)
                mass_prototype_i = tf.expand_dims(mass_prototype_i, -2)
                mass_prototype = tf.concat([mass_prototype, mass_prototype_i], -2)
        return mass_prototype

    def get_config(self):
        config = super(DS2, self).get_config()
        config.update({'num_class': self.num_class})
        return config


class DS2_omega(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DS2_omega, self).__init__(**kwargs)

    def call(self, inputs):
        mass_omega_sum = tf.reduce_sum(inputs, -1, keepdims=True)
        mass_omega_sum = tf.subtract(1., mass_omega_sum, name=None)
        mass_with_omega = tf.concat([inputs, mass_omega_sum], -1)
        return mass_with_omega

    def get_config(self):
        config = super(DS2_omega, self).get_config()
        return config


class DS3_Dempster(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DS3_Dempster, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_dim = input_shape[-2]

    def call(self, inputs):
        m1 = inputs[:, :, :, 0, :]
        omega1 = tf.expand_dims(inputs[:, :, :, 0, -1], -1)
        for i in range(self.input_dim - 1):
            m2 = inputs[:, :, :, (i + 1), :]
            omega2 = tf.expand_dims(inputs[:, :, :, (i + 1), -1], -1)
            combine1 = tf.multiply(m1, m2, name=None)
            combine2 = tf.multiply(m1, omega2, name=None)
            combine3 = tf.multiply(omega1, m2, name=None)
            combine1_2 = tf.add(combine1, combine2, name=None)
            combine2_3 = tf.add(combine1_2, combine3, name=None)
            combine2_3_omega = tf.divide(combine2_3[:, :, :, -1], 3)
            combine2_3_omega = tf.expand_dims(combine2_3_omega, -1)
            combine2_3 = tf.concat([combine2_3[:, :, :, 0:-1], combine2_3_omega], -1)
            m1 = combine2_3
            omega1 = tf.expand_dims(combine2_3[:, :, :, -1], -1)
        return m1

    def get_config(self):
        config = super(DS3_Dempster, self).get_config()
        return config


class DS3_normalize(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DS3_normalize, self).__init__(**kwargs)

    def call(self, inputs):
        mass_combine_normalize = inputs / tf.reduce_sum(inputs, axis=-1, keepdims=True)
        return mass_combine_normalize

    def get_config(self):
        config = super(DS3_normalize, self).get_config()
        return config
class SelectSingleton(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SelectSingleton, self).__init__(**kwargs)
        
    def call(self, inputs):
        mass_class = inputs[:, :, :, :-1]  # Exclude the last dimension (omega) which is not part of the output
        return mass_class
    
    def get_config(self):
        config = super(SelectSingleton, self).get_config()
        return config