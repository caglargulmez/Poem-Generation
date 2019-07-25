import dynet as dy
import json
import math
from numpy import argmax
from keras.utils import to_categorical
from itertools import compress
from numpy.random import multinomial



data_dir = './input/unim_poem.json'
# load json file
def load_json():
    with open(data_dir, "r") as read_file:
        return json.load(read_file)

data = load_json()

# loads bigrams of corpus
def load_bigrams_tuple(corpus):
    bigrams = []
    for line in corpus:
        words = line.split(' ')
        for idx in range(len(words)-1):
            bigrams.append((words[idx], words[idx+1]))

    return bigrams

# loads list of bigram couples
def load_bigrams_list(corpus):
    bigrams = []
    for line in corpus:
        words = line.split(' ')
        for idx in range(len(words)-1):
            bigrams.append(words[idx] + ' ' + words[idx+1])

    return bigrams

# loads list of unique words
def load_unigrams_list(corpus):
    return list({word for line in corpus for word in line.split(' ') if word != None}) #not in [None, 'the', 'a','i','no','in','and', 'for', 'of', 'to', 'that']})

# loads list of bigrams with tokens <s> </s>
def load_list_tokens(data):

    flag = 1
    bigrams = []
    counter = 0


    for element in data:

        lines = element['poem'].split('\n')

        for line in lines:
            counter +=1
            line = line.split(' ')
            if flag==1:
                bigrams.append('<s>' + ' '  + line[0])
                flag = 0
            for idx in range(len(line) - 1):
                bigrams.append(line[idx] + ' ' + line[idx+1])

            if counter == len(lines):
                bigrams.append(line[-1] +' ' +  '</s>')
            else:
                bigrams.append(line[len(line) - 1] + ' '+  '\n')


        flag = 1

    return bigrams

# weightedChoice with numpy.multinomials
def weightedChoice(weights, objects):
    """Return a random item from objects, with the weighting defined by weights
    (which must sum to 1)."""
    return next(compress(objects, multinomial(1, weights, 1)[0]))

#calculating perplexity of probabilities
def calc_perplexity(list):
    total_probs = 0
    for prob in list:
        total_probs += math.log2(prob)

    return 1 / math.pow(2, (total_probs/len(list)))

print('###########  Program is started. ##########')



# CORPUS LOADING
corpus = [line for element in data for line in element['poem'].split('\n')]
print('###########  Corpus is created.  ###########')
print(corpus[0:10])
print('\tLength of corpus: {}'.format(len(corpus)))

# BIGRAMS LOADING
bigrams= load_list_tokens(data)
print('###########  Bigrams is created.  ###########')
print(bigrams[0:100])
print('\tLength of bigrams:', len(bigrams))

# UNIGRAMS LOADING
unigrams = load_unigrams_list(corpus=corpus)
unigrams.append('<s>')    # add starting poem token
unigrams.append('</s>')   # add end poem token
unigrams.append('\n')
print('###########  Unigrams are created.  ###########')
print(unigrams[0:10])
print('###########  Length of Unigrams: {} ###########'.format(len(unigrams)))

# FIRST INDEX IS '' so avoid this word
unigrams = unigrams[1:]

# Indexes to generate one hot vector
indexes = [i for i in range(len(unigrams))]

# One hot encoded vectors
one_hot_encoded = to_categorical(indexes)


# k,v -> word:index dictionary
word_index = {}
# k,v -> index:word dictionary
index_word = {}

for i in range(len(unigrams)):
    word_index[unigrams[argmax(one_hot_encoded[i])]] = i
    index_word[i] = unigrams[argmax(one_hot_encoded[i])]

print('###########  Dictionaries which named word_index & index_word are created  ###########')


# create list of tuples which holds two indexes of bigram couples
data = []
for bigram in bigrams:
    bigram2 = bigram.split(' ')
    if bigram2[0] != '' and bigram2[1] != '':
        if bigram2[0] != None and bigram2[1] != None:
        #if bigram2[0] not in [None, 'the', 'a','i','no','in','and', 'for', 'of', 'to', 'that', '\n'] and bigram2[1] not in [None, 'the', 'a','i','no','in','and', 'for', 'of', 'to', 'that', '\n']:
            data.append((word_index[bigram2[0]], word_index[bigram2[1]]))


# Dynet model
model = dy.Model()
pW = model.add_parameters((150,115562))
pb = model.add_parameters(150)
pU = model.add_parameters((115562, 150))
pd = model.add_parameters(115562)

trainer = dy.SimpleSGDTrainer(model)

EPOCHS = 100


for epoch in range (EPOCHS):
    epoch_loss = 0.0
    for x,y in data[0:115562]:
        dy.renew_cg()

        W = dy.parameter(pW)
        b = dy.parameter(pb)
        U = dy.parameter(pU)
        d = dy.parameter(pd)


        x_val = dy.inputVector(list(one_hot_encoded[x]))
        h_val = dy.tanh(W * x_val + b)

        y_val = U * h_val + d

        loss = dy.pickneglogsoftmax(y_val, y)
        epoch_loss += loss.scalar_value()

        loss.backward()
        trainer.update()

    print('Epoch', epoch, '. loss =', epoch_loss/115562)


prob_list = []
def generate_poem():
    start = '<s>'
    poem = ''

    wordflag = 0

    for i in range(25):
        dy.renew_cg()

        W = dy.parameter(pW)
        b = dy.parameter(pb)
        U = dy.parameter(pU)
        d = dy.parameter(pd)

        x_val = dy.inputVector(list(one_hot_encoded[word_index[start]]))
        h_val = dy.tanh(W * x_val + b)

        y_val = U * h_val + d

        probs = dy.softmax(y_val)

        poem += start

        poem += ' '
        wordflag += 1

        if wordflag == 5:
            poem += '\n'
            wordflag = 0

        start = weightedChoice(probs.value(), unigrams)
        prob_list.append(probs.__getitem__(word_index[start]).value())
    return poem


poem = generate_poem()
perp = calc_perplexity(prob_list)
