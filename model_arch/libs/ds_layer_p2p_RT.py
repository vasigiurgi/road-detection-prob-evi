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
    
    
# Belief DBI

import tensorflow as tf
from tensorflow import keras

# Belief-Plausibility joint calculation of BBAs, m(.)s   
# m(.)s have only singleton plus omega as focal elements 
class BeliefPlausibility(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BeliefPlausibility, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.cardinal_fod = input_shape[-1] - 1  # cardinality of frame of discernment (fod)
        
    @tf.function(autograph=True)         
    def call(self, inputs):
        zero_value = tf.zeros_like(inputs)[:, :, :, 1]
        zero_value = tf.expand_dims(zero_value, axis=-1)
        unity_value = tf.ones_like(zero_value)
        
        singleton_index = tf.range(0, self.cardinal_fod)
        singleton_index = tf.math.pow(2, singleton_index)
        
        bel = tf.zeros_like(unity_value)
        
        for i in range(pow(2, self.cardinal_fod) - 2):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(bel, tf.TensorShape([None, 384, 1248, None]))]
            )
            if i == 0:                            
                index = tf.bitwise.bitwise_and(i + 1, singleton_index)
                index = tf.cast(index, tf.float32)
                index = tf.math.divide(tf.math.log(index), tf.math.log(2.0))
                mask = ~tf.math.is_inf(index)
                index = tf.boolean_mask(index, mask)
                index = tf.cast(index, tf.int32)
                bel_i = tf.gather(inputs, index, axis=-1)
                bel = tf.reduce_sum(bel_i, axis=-1, keepdims=True)
            if i >= 1:
                index = tf.bitwise.bitwise_and(i + 1, singleton_index)
                index = tf.cast(index, tf.float32)
                index = tf.math.divide(tf.math.log(index), tf.math.log(2.0))
                mask = ~tf.math.is_inf(index)
                index = tf.boolean_mask(index, mask)
                index = tf.cast(index, tf.int32)
                bel_i = tf.gather(inputs, index, axis=-1)
                bel_i = tf.reduce_sum(bel_i, axis=-1, keepdims=True)
                bel = tf.concat([bel, bel_i], axis=-1)

        pl = bel        
        bel = tf.concat([zero_value, bel, unity_value], -1)
        mass_omega = tf.expand_dims(inputs[:, :, :, -1], axis=-1)
        pl = tf.math.add(pl, mass_omega)
        pl = tf.concat([zero_value, pl, unity_value], axis=-1)
            
        return [bel, pl]
        
    def get_config(self):
        config = super(BeliefPlausibility, self).get_config()
        return config

# Belief-Plausibility joint calculation of focused bba/ logical bba m_x(.)s    
# The focused bba is focused on a non empty subset of Omega, fod
class BeliefPlausibilityFocused(tf.keras.layers.Layer):
    def __init__(self, focal, **kwargs):
        super(BeliefPlausibilityFocused, self).__init__(**kwargs)
        self.focal = focal
    
    def build(self, input_shape):
        self.cardinal_fod = input_shape[-1] - 1  # cardinality of frame of discernment (fod)
    
    @tf.function(autograph=True)          
    def call(self, inputs):
        bel = tf.zeros_like(inputs)[:, :, :, 1]
        bel = tf.expand_dims(bel, axis=-1)
        pl = tf.zeros_like(bel)
        
        zero_value = tf.zeros_like(bel)
        unity_value = tf.ones_like(bel)
        
        for i in range(pow(2, self.cardinal_fod) - 1):
            tf.autograph.experimental.set_loop_options(
                shape_invariants=[(bel, tf.TensorShape([None, 384, 1248, None])), (pl, tf.TensorShape([None, 384, 1248, None]))]
            )
            index_contain = tf.bitwise.bitwise_and(i + 1, self.focal)
            
            bel = tf.cond(index_contain == self.focal, 
                          lambda: tf.concat([bel, unity_value], -1), 
                          lambda: tf.concat([bel, zero_value], -1)
                          )
            
            pl = tf.cond(index_contain > 0, 
                          lambda: tf.concat([pl, unity_value], -1), 
                          lambda: tf.concat([pl, zero_value], -1)
                          )
            
        return [bel, pl]
    
    def get_config(self):
        config = super(BeliefPlausibilityFocused, self).get_config()
        config.update({
            'focal': self.focal
        })
        return config

# Wassertein's distance square
class Wassertein(tf.keras.layers.Layer):
    def __init__(self, focal, **kwargs):
        super(Wassertein, self).__init__(**kwargs)
        self.focal = focal
        self.belief_plausibility_bba = BeliefPlausibility()     
        self.belief_plausibility_focused = BeliefPlausibilityFocused(self.focal)
    
    def call(self, inputs):
        bel_pl_bba = self.belief_plausibility_bba(inputs)
        bel_pl_x = self.belief_plausibility_focused(inputs)
        sum_bba = tf.math.add(bel_pl_bba[0], bel_pl_bba[1]) * 0.5
        dif_bba = tf.math.subtract(bel_pl_bba[1], bel_pl_bba[0]) * 0.5
        sum_x = tf.math.add(bel_pl_x[0], bel_pl_x[1]) * 0.5
        dif_x = tf.math.subtract(bel_pl_x[1], bel_pl_x[0]) * 0.5
        
        # difference of sum
        dif_sum = tf.math.subtract(sum_bba, sum_x)
        dif_sum = tf.math.pow(dif_sum, 2)
        
        # difference of difference
        dif_dif = tf.math.subtract(dif_bba, dif_x)
        dif_dif = tf.math.pow(dif_dif, 2)
        dif_dif = tf.math.divide(dif_dif, 3)
        
        distance_wassertein = dif_sum + dif_dif  # Wassertein's distance square
        
        return distance_wassertein
    
    def get_config(self):
        config = super(Wassertein, self).get_config()
        config.update({
            'focal': self.focal
        })
        return config

# belief interval distance
class BeliefIntervalDistance(tf.keras.layers.Layer):
    def __init__(self, space, **kwargs):
        super(BeliefIntervalDistance, self).__init__(**kwargs)
        self.space = space
    
    def build(self, input_shape):
        self.cardinal_fod = input_shape[-1] - 1  # cardinality of frame of discernment (fod)
             
    def call(self, inputs):
        norm_const = tf.math.pow(2, self.cardinal_fod - 1)
        norm_const = tf.cast(norm_const, tf.float32)
        for i in self.space:
            dw = Wassertein(i)(inputs)
            if i == self.space[0]:
                dBI = tf.math.reduce_sum(dw, axis=-1)
                dBI = tf.math.divide(dBI, norm_const)
                dBI = tf.math.sqrt(dBI)
                dBI = tf.expand_dims(dBI, axis=-1)
            else:
                dBI_i = tf.math.reduce_sum(dw, axis=-1)
                dBI_i = tf.math.divide(dBI_i, norm_const)
                dBI_i = tf.math.sqrt(dBI_i)
                dBI_i = tf.expand_dims(dBI_i, axis=-1)
                dBI = tf.concat([dBI, dBI_i], -1)
        return dBI
    
    def get_config(self):
        config = super(BeliefIntervalDistance, self).get_config()
        config.update({
            'space': self.space
        })
        return config
