import pickle
from keras.models import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model

def class_discriminator(n_classes):
    in_gaussian = Input(shape=(2,))
    inp = Dense(100, activation='relu')(in_gaussian)
    inp = Dense(50, activation='relu')(inp)
    inp = Dense(25, activation='relu')(inp)
    out1 = Dense(1, activation='sigmoid')(inp)
    out2 = Dense(n_classes)(inp)
    model = Model(in_gaussian, [out1, out2])
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['binary_crossentropy', 'sparse_categorical_crossentropy'], optimizer=opt)
    return model

model = class_discriminator(15)
model.summary()
plot_model(model, to_file='traffic_rate_discriminator_plot.png', show_shapes=True, show_layer_names=True)
