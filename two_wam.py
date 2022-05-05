import sys
sys.path.append(".")
from tensorflow.keras.layers import Layer
from tensorflow.keras import backend



class Two_WAM(Layer):

    def __init__(self, **kwargs):
        super(Two_WAM, self).__init__(**kwargs)


    def build(self, input_shape):
        # assert isinstance(input_shape, list)
        # Create a trainable weight variable for this layer.

        def value(shape):
            if type(shape) is int:
                return shape
            else:
                return shape.value


        self.linear = self.add_weight(name='kernel',
                                      shape=(1,value(input_shape[3])),
                                      initializer='uniform',
                                      trainable=True)
        self.exp = self.add_weight(name='exp',
                                       shape=(1,value(input_shape[3])),
                                       initializer='uniform',
                                       trainable=True)
        super(Two_WAM, self).build(input_shape)  # Be sure to call this at the end


    def call(self, x):

        W = 10**self.exp[0]*self.linear[0]
        X0 = x * W / backend.sum(W)
        X_aux = backend.sum(X0,3,keepdims=True)

        X_aux2 = x * X_aux

        return X_aux2, X_aux


    def compute_output_shape(self, input_shape):
        # assert isinstance(input_shape, list)
        batch, shape_a, shape_b, shape_c = input_shape

        return (batch, shape_a, shape_b, 1)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
        })
        return config
