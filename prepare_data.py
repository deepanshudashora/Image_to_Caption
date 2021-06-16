from feature_engineering import load_set
from feature_engineering import load_clean_descriptions
from feature_engineering import load_photo_features
from feature_engineering import create_tokenizer
from feature_engineering import max_length
from feature_engineering import create_sequences

from features_image import extract_features
from pickle import dump

from text_clean import load_doc
from text_clean import load_descriptions
from text_clean import clean_descriptions
from text_clean import to_vocabulary
from text_clean import save_descriptions


def prepare_the_data(image_path,text_path):
    directory = image_path
    features = extract_features(directory)
    print('Extracted Features: %d' % len(features))
    # save to file
    dump(features, open('features.pkl', 'wb'))


    # clean text data
    filename = text_path
    doc = load_doc(filename)
    descriptions = load_descriptions(doc)
    print('Loaded: %d ' % len(descriptions))
    clean_descriptions(descriptions)
    vocabulary = to_vocabulary(descriptions)
    print('Vocabulary Size: %d' % len(vocabulary))
    save_descriptions(descriptions, 'text_descriptions.txt')


    filename = text_path
    train = load_set(filename)
    print('Dataset: %d' % len(train))
    # descriptions
    train_descriptions = load_clean_descriptions('text_descriptions.txt', train)
    print('Descriptions: train=%d' % len(train_descriptions))
    # photo features
    train_features = load_photo_features('features.pkl', train)
    print('Photos: train=%d' % len(train_features))

    # prepare tokenizer
    tokenizer = create_tokenizer(train_descriptions)
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)

    maximum_length = max_length(train_descriptions)
    print('Description Length: %d' % maximum_length)
    # prepare sequences
    X1, X2, y = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)
    return X1,X2,y
