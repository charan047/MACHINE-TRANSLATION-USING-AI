# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
import tensorflow
print(tensorflow.__version__)

project_path ="/content/drive/MyDrive/mt2/"

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import string
from string import digits
import re
import os
from numpy import array, argmax, random, take
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, RepeatVector,TimeDistributed,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint,EarlyStopping,ReduceLROnPlateau
from keras.models import load_model
from tensorflow import keras
from keras import optimizers
import matplotlib.pyplot as plt
# %matplotlib inline

# pd.set_option('display.max_colwidth', 200)

english_sentances = []
telugu_sentances = []
with open(project_path+"/english_telugu_data.txt", mode='rt', encoding='utf-8') as fp:
    for line in fp.readlines():
        eng_tel = line.split("++++$++++")
        english_sentances.append(eng_tel[0])
        telugu_sentances.append(eng_tel[1])

data = pd.DataFrame({"english_sentances":english_sentances,"telugu_sentances":telugu_sentances})

data.head()

data.shape

# Let's take 70000 phrases from data
data = data.iloc[:155000,:]

contraction_mapping = {"ain't": "is not", "aren't": "are not","can't": "cannot", "'cause": "because", "could've": "could have", "couldn't": "could not",
                           "didn't": "did not",  "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not", "haven't": "have not",
                           "he'd": "he would","he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y": "how do you", "how'll": "how will", "how's": "how is",
                           "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have","I'm": "I am", "I've": "I have", "i'd": "i would",
                           "i'd've": "i would have", "i'll": "i will",  "i'll've": "i will have","i'm": "i am", "i've": "i have", "isn't": "is not", "it'd": "it would",
                           "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have","it's": "it is", "let's": "let us", "ma'am": "madam",
                           "mayn't": "may not", "might've": "might have","mightn't": "might not","mightn't've": "might not have", "must've": "must have",
                           "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not", "needn't've": "need not have","o'clock": "of the clock",
                           "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not", "sha'n't": "shall not", "shan't've": "shall not have",
                           "she'd": "she would", "she'd've": "she would have", "she'll": "she will", "she'll've": "she will have", "she's": "she is",
                           "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have", "so've": "so have","so's": "so as",
                           "this's": "this is","that'd": "that would", "that'd've": "that would have", "that's": "that is", "there'd": "there would",
                           "there'd've": "there would have", "there's": "there is", "here's": "here is","they'd": "they would", "they'd've": "they would have",
                           "they'll": "they will", "they'll've": "they will have", "they're": "they are", "they've": "they have", "to've": "to have",
                           "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will", "we'll've": "we will have", "we're": "we are",
                           "we've": "we have", "weren't": "were not", "what'll": "what will", "what'll've": "what will have", "what're": "what are",
                           "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have", "where'd": "where did", "where's": "where is",
                           "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is", "who've": "who have",
                           "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not", "won't've": "will not have",
                           "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have", "y'all": "you all",
                           "y'all'd": "you all would","y'all'd've": "you all would have","y'all're": "you all are","y'all've": "you all have",
                           "you'd": "you would", "you'd've": "you would have", "you'll": "you will", "you'll've": "you will have",
                           "you're": "you are", "you've": "you have"}

# clean english sentances
def clean_eng(text):
    # Lowercase all characters
    text = text.lower()
    # map contractions
    text = ' '.join([contraction_mapping[w] if w in contraction_mapping else w for w in text.split(" ")])
    # Remove quotes
    text = re.sub("'", '', text)
    # Remove all the special characters
    exclude = set(string.punctuation) # Set of all special characters
    text = ''.join([c for c in text if c not in exclude])
    # Remove all numbers from text
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    # Remove extra spaces
    text= text.strip()

    return text 

# clean telugu sentances
def clean_tel(text):
    # Lowercase all characters
    text = text.lower()
    # Remove quotes
    text = re.sub("'", '', text)
    # Remove all the special characters
    exclude = set(string.punctuation)  # Set of all special characters
    text = ''.join([c for c in text if c not in exclude])
    # Remove all numbers from text
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    # Remove Telugu numbers  from text
    text = re.sub("[౦౧౨౩౪౫౬౭౮౯]", '', text)
    # Remove extra spaces
    text= text.strip()

    return text

# clean text
data_df = data.copy()
data_df["english_sentances"] = data_df["english_sentances"] .apply(lambda x: clean_eng(x))
data_df["telugu_sentances"] = data_df["telugu_sentances"] .apply(lambda x: clean_tel(x))

data_df.head()

# empty lists
eng_l = []
tel_l = []
# populate the lists with sentence lengths
for i in data_df["english_sentances"].values:
      eng_l.append(len(i.split()))

for i in data_df["telugu_sentances"].values:
      tel_l.append(len(i.split()))

length_df = pd.DataFrame({'eng':eng_l, 'tel':tel_l})

length_df.hist(bins = 50)
plt.show()

# function to build a tokenizer
def tokenization(lines):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

# prepare english tokenizer
eng_tokenizer = tokenization(data_df["english_sentances"])
eng_vocab_size = len(eng_tokenizer.word_index) + 1

eng_length = 43
print('English Vocabulary Size: %d' % eng_vocab_size)

# prepare Telugu  tokenizer
tel_tokenizer = tokenization(data_df["telugu_sentances"])
tel_vocab_size = len(tel_tokenizer.word_index) + 1

tel_length = 26
print('Telugu Vocabulary Size: %d' % tel_vocab_size)

# encode and  pad sequences
def encode_sequences(tokenizer, length, lines):
    # integer encode sequences
    seq = tokenizer.texts_to_sequences(lines)
    # pad sequences with 0 values
    seq = pad_sequences(seq, maxlen=length, padding='post')
    return seq

# split data into train and test set
train, test = train_test_split(data_df, test_size=0.2, random_state = 12)

# prepare training data
trainX = encode_sequences(eng_tokenizer, eng_length, train["english_sentances"])
trainY = encode_sequences(tel_tokenizer, tel_length, train["telugu_sentances"])

# prepare validation data
testX = encode_sequences(eng_tokenizer, eng_length, test["english_sentances"])
testY = encode_sequences(tel_tokenizer, tel_length, test["telugu_sentances"])

trainX.shape,trainY.shape,testX.shape,testY.shape

# data generator, intended to be used in a call to model.fit_generator()
# def data_generator (eng_tokenizer,tel_tokenizer, eng_length,tel_length, eng_data,tel_data) :
#     # loop for each sentance
#     while 1 :

#         for eng_sentance,tel_sentance in zip(eng_data,tel_data):
#             # integer encode sequences
#             eng_seq = eng_tokenizer.texts_to_sequences(eng_sentance)
#             tel_seq = tel_tokenizer.texts_to_sequences(tel_sentance)
#             # pad sequences with 0 values
#             eng_seq = pad_sequences(eng_seq, maxlen=eng_length, padding='post')
#             tel_seq = pad_sequences(tel_seq, maxlen=tel_length, padding='post')
#             yield eng_seq,tel_seq


# build NMT model
def define_model(in_vocab,out_vocab, in_timesteps,out_timesteps,units):
    model = Sequential()
    model.add(Embedding(in_vocab, units, input_length=in_timesteps, mask_zero=True))
    model.add(LSTM(units))
    model.add(Dropout(0.2))
    model.add(RepeatVector(out_timesteps))
    model.add(LSTM(units, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(out_vocab, activation='softmax')))
    return model

# model compilation
model = define_model(eng_vocab_size,tel_vocab_size,eng_length,tel_length, 512)
model.summary()

rms = optimizers.RMSprop()
model.compile(optimizer=rms, loss='sparse_categorical_crossentropy')

# Defining a helper function to save the model after each epoch 
# in which the loss decreases 

filepath = project_path+'NMT_model.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# Defining a helper function to reduce the learning rate each time 
# the learning plateaus 
reduce_alpha = ReduceLROnPlateau(monitor ='val_loss', factor = 0.2, patience = 1, min_lr = 0.001)
# stop traning if there increase in loss
# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=2)
callbacks = [checkpoint, reduce_alpha]

# train the model
# epochs = 30
# train_steps = len(train["english_sentances"])
# val_steps = len(test["english_sentances"])
# create the data generator
# prepare training data
#train_gen = data_generator(eng_tokenizer,tel_tokenizer, eng_length,tel_length, train["english_sentances"],train["telugu_sentances"])
# prepare validation data
#test_gen = data_generator(eng_tokenizer,tel_tokenizer, eng_length,tel_length, test["english_sentances"],test["telugu_sentances"])

# model.fit_generator(train_gen, epochs=epochs, steps_per_epoch=train_steps,validation_data=(test_gen),validation_steps=val_steps,callbacks=callbacks, verbose=1)
# save model
# model.save(project_path+'model_img_cap_pad.h5')

# train model
history = model.fit(trainX, trainY.reshape(trainY.shape[0], trainY.shape[1], 1),
                    epochs=55, batch_size=128, validation_split = 0.2,callbacks=callbacks, verbose=1)

# save model
model.save(project_path+'NMT_model1')

# reconstructed_model = keras.models.load_model("my_model")

# convert the history.history dict to a pandas DataFrame:     
# hist_df = pd.DataFrame(history.history) 
# save to  json:  
# hist_json_file = project_path+'history.json' 
# with open(hist_json_file, mode='w') as f:
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train','validation'])
#     hist_df.to_json(f)
plt.show()


# get 10 random ids of test samples
idx = random.randint(testX.shape[0], size=10)
# get 10 encoded english test samples
encoded_english_actual = testX[idx,:]
# get 10 actual english sentences 
eng_actual = test["english_sentances"].values
eng_actual = eng_actual[idx]
# get 10 actual telugu sentences
actual = test["telugu_sentances"].values

actual = actual[idx]

# load model weights
# model = keras.models.load_model("/content/drive/MyDrive/mtNMT_model.h5")
# predict english sentence to telugu sentence
preds = model.predict_classes(encoded_english_actual.reshape((encoded_english_actual.shape[0],encoded_english_actual.shape[1])))

def get_word(n, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == n:
            return word
    return None


preds_text = []
for i in preds:
    temp = []
    for j in range(len(i)):
        t = get_word(i[j], tel_tokenizer)
        
        if j > 0:
            if (t == get_word(i[j-1], tel_tokenizer)) or (t == None):
                temp.append('')
            else:
                temp.append(t)
        else:
            if(t == None):
                temp.append('')
            else:
                temp.append(t) 

    preds_text.append(' '.join(temp))

pred_df = pd.DataFrame({'english_actual':eng_actual, 'telugu_actual' : actual, 'telugu_predicted' : preds_text})

from google.colab import data_table
from vega_datasets import data

data_table.enable_dataframe_formatter()
# print 10 rows
pred_df.tail(15)

# convert the history.history dict to a pandas DataFrame:     
hist_df = pd.DataFrame(history.history) 

# save to json:  
hist_json_file = project_path+'history.json' 
with open(hist_json_file, mode='w') as f:
    hist_df.to_json(f)
    
#this cell is to plot the loss & val_loss graph
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#this cell is to plot the train acc & val_acc graph
acc = history.history['acc']
val_acc = history.history['val_acc']
plt.plot(epochs, acc, 'y', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
