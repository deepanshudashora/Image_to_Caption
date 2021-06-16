from prepare_data import prepare_the_data
from model import caption_model
from keras.callbacks import ModelCheckpoint

x1_train, x2_train, y_train,vocab_size_train,maximum_length_train = prepare_the_data("Data/training_Images","Data/training_captions.txt")
x1_test , x2_test, y_test,vocab_size_test,maximum_length_test = prepare_the_data("Data/testing_images","Data/testing_caption.txt")

model = caption_model(vocab_size_train, maximum_length_train)


# define checkpoint callback
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# fit model
model.fit([x1_train, x2_train], y_train, epochs=20, verbose=2, callbacks=[checkpoint], validation_data=([x1_test, x2_test], y_test))

