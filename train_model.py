from prepare_data import prepare_the_data
from model import caption_model
from feature_engineering import data_generator

train_descriptions, train_features, tokenizer, maximum_length, vocab_size = prepare_the_data("Data/training_Images","Data/training_captions.txt")
valid_description, valid_features, valid_tokenizer, valid_maximum_length, valid_vocab_size = prepare_the_data("Data/testing_images","Data/testing_caption.txt")


# define the model
model = caption_model(vocab_size, maximum_length)
# train the model, run epochs manually and save after each epoch
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    # create the data generator
    train_generator = data_generator(train_descriptions, train_features, tokenizer, maximum_length, vocab_size)
    valid_generator = data_generator(valid_description, valid_features, valid_tokenizer, valid_maximum_length, valid_vocab_size)
    # fit for one epoch
    model.fit(train_generator, epochs=1, steps_per_epoch=steps, verbose=1, validation_data=valid_generator)
    # save model
    print("Model is saving")
    model.save("trained_models/"+'model_' + str(i) + '.h5')