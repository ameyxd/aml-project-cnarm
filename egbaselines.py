import pandas as pd
import tensorflow as tf
from tensorflow.python.keras.models import Sequential, Model
from tensorflow.python.keras.layers import GRU, Dense, Input
from tensorflow.python.keras.layers import Embedding
from keras.utils.np_utils import to_categorical
from tensorflow.python.keras.metrics import top_k_categorical_accuracy
import numpy as np
import keras.backend as k

tf.enable_eager_execution()

#x and c are individual sessions
integratecontext = []
def concat(x,c):
    for i in range(max_length):
        integratecontext.append(T.concatenate([x[i],c[i]], axis = 0))


def tkca(y_true,y_pred):
    # last_good_index = max([i for i in range(19) if int(tfksum(y_true[i,:]))!=0])
    mc = 0.0
    for i in range(19):
        print(y_true)
        print(y_pred)
        mc = mc + top_k_categorical_accuracy(y_true[:,i,:],y_pred[:,i,:])
        if y_true.numpy()[0][i][0] == 1:
            return mc/i

df = pd.read_csv("fprod.dat", delimiter=" ", header= None)
dfprod = df.loc[:,1::2]
dfcont = df.loc[:,2::2]
dfprod = dfprod.values
dfcont = dfcont.values

xp = dfprod[:,0:-1]
yp = dfprod[:,1:]
xc = dfcont[:,0:-1]
yc = dfcont[:,1:]

z = 1000000

xptrain  =  xp[0:z,:]
yptrain = yp[0:z,:]
xptest = xp[z:,:]
yptest = yp[z:,:]

vocab = np.max(dfprod)+1
inp = Input(shape=(19,))
emb = Embedding(input_dim=vocab,mask_zero=True,output_dim= 100, input_length=len(xp[0])) (inp)
g = GRU(units=100, return_sequences= True) (emb)
d = Dense(vocab, activation='softmax')(g)
model = Model(inputs=inp, outputs=d)
optimizeradam = tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy', optimizer=optimizeradam, metrics= [tkca])
categorical_labels = to_categorical(yptrain)
categorical_labels2 = to_categorical(yptest)
model.fit(xptrain,categorical_labels, epochs= 5, batch_size= 1)
print(model.evaluate(xptest,categorical_labels2))

# model = Sequential()
# model.add(Embedding(input_dim=vocab,mask_zero=True,output_dim= 100, input_length=len(xp[0])))
# model.add(GRU(units=100, return_sequences= True))
# model.add(Dense(vocab, activation='softmax'))
# categorical_labels = to_categorical(yp)
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics= ['acc'])
# model.summary()