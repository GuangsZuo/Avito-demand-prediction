
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import keras as k
import matplotlib.pyplot as plt

import os
import gc
os.environ["CUDA_VISIBLE_DEVICES"]="3"


# In[2]:


train = pd.read_csv("train.csv",parse_dates=["activation_date"])
test = pd.read_csv("test.csv",parse_dates=["activation_date"])
del train["image"]
aggregated_features = pd.read_csv("aggregated_features.csv")


# In[3]:


train["text"] = train["title"].astype(str) + "," + train["description"].astype(str) #+ "," \
#+ train["param_1"].astype(str) + "," + train["param_2"].astype(str) + "," + train["param_3"].astype(str)

test["text"] = test["title"].astype(str) + "," + test["description"].astype(str) # + "," \
#+ test["param_1"].astype(str) + "," + test["param_2"].astype(str) + "," + test["param_3"].astype(str)
train["text"] = train["text"].astype(str)
test["text"] =test["text"].astype(str)


# In[4]:


is_first = 1

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
max_features = 200000
embed_size = 300
maxlen = 150

if is_first:

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(pd.concat((train['text'],test['text'])))
    train_words = tokenizer.texts_to_sequences(train['text'])
    test_words = tokenizer.texts_to_sequences(test['text'])
    train_words = pad_sequences(train_words, maxlen=maxlen)
    test_words = pad_sequences(test_words, maxlen=maxlen)
    print(len(list(tokenizer.word_index)))
    
    embeding_file_path_1 = "./cc.ru.300.vec"
    embeding_file_path_2 = "/home/LAB/zuogs/crawl-300d-2M.vec"
    def get_coef(word, *coefs):
        return word, np.asarray(coefs, dtype=np.float32)
    embeding_dict = dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path_1))
    embeding_dict.update(dict(get_coef(*s.strip().split(" ")) for s in open(embeding_file_path_2)))
    word_index = tokenizer.word_index
    max_words = min(max_features, len(word_index)) + 1 
    embeding_matrix = np.zeros((max_words, embed_size))
    lose = 0
    lost_words = []
    for word,i in word_index.items():
        if word not in embeding_dict: 
            lose += 1
            word = "something"#"something"
        if i>=max_words: 
            continue 
        embeding_matrix[i] = embeding_dict[word]
    print(lose)
    del embeding_dict
    gc.collect()
    
    np.save("embeding-300d-fasttext-withoutparam",embeding_matrix)
    np.save("train_words-withoutparam",train_words)
    np.save("test_words-withoutparam",test_words)
else:
    embeding_matrix = np.load("embeding-300d-fasttext-withoutparam.npy")
    train_words = np.load("train_words-withoutparam.npy")
    test_words = np.load("test_words-withoutparam.npy")
    max_words = embeding_matrix.shape[0]


# In[5]:


embeding_matrix.shape


# In[6]:


features = ["price", "item_seq_number"] 
categorical_features = []
df = pd.concat([train,test], axis=0)

################################ nan encoding 
nan_features = [ "price", "param_1", "param_2", "param_3", "description"] # others are useless
for f in nan_features:
    feature_name = f + "is_nan"
    df[feature_name] = df[f].isnull().astype(int)
    if f == "price": features.append(feature_name)
gc.collect()

###############################filling NAN
df["image_top_1"] = df["image_top_1"].fillna(df["image_top_1"].max()+1)
df["param_1"] = df["param_1"].fillna("missing")
df["param_2"] = df["param_2"].fillna("missing")
df["param_3"] = df["param_3"].fillna("missing")
temp = df[["category_name", "price"]].groupby("category_name")["price"].median().reset_index().rename(columns={"price": "category_median_price"})
df = df.merge(temp, how="left", on="category_name")
df["price"] = df["price"].fillna(df["category_median_price"])

fs = ["param_1", "param_2", "param_3", "image_top_1", "price"]
train[fs] = df[fs][:train.shape[0]]

df["price"] = np.log1p(df["price"])

############################### user_id_count

features_to_count = ["user_id"] # others are useless 
for f in features_to_count:
    feature_name = f + "_count"
    temp = df[[f,"price"]].groupby([f])["price"].count().reset_index().rename(columns={"price": feature_name})
    df = df.merge(temp, how="left", on=[f])
    features.append(feature_name)
    del temp
gc.collect()

############################### weekday 

df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["dayofyear"] = df['activation_date'].dt.dayofyear
categorical_features.extend(["Weekday"])

############################### label encoding
categorical_features_tobe = [ "region", "city", "category_name" ,"user_type","param_1","param_2","param_3","image_top_1"] 
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for cf in categorical_features_tobe:
    le.fit(df[cf].astype(str))
    le_feature_name = "{}_le".format(cf)
    df[le_feature_name] = le.transform(df[cf].astype(str))
    categorical_features.append(le_feature_name)
del le 
gc.collect()

############################## text feature 
textfeats = ["description", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) 
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 
    features.extend([cols + '_num_words',cols + '_num_unique_words', cols + '_words_vs_unique'])
    
############################## aggregate features
df = df.merge(aggregated_features, how="left", on="user_id")
features.extend(aggregated_features.columns[1:])
df[aggregated_features.columns[1:]] = df[aggregated_features.columns[1:]].fillna(df[aggregated_features.columns[1:]].mean())


######################################### user features

feature_name="user_mean_price"
temp = df[["user_id","price"]].groupby(['user_id'])["price"].mean().reset_index().rename(columns={"price": feature_name})
df = df.merge(temp, how="left", on=["user_id"])
features.append(feature_name)
del temp
gc.collect()

feature_name ="user_mean_description_length"
temp = df[["user_id","description_num_words"]].groupby(['user_id'])["description_num_words"].mean().reset_index().rename(columns={"description_num_words": feature_name})
df = df.merge(temp, how="left", on=["user_id"])
features.append(feature_name)
del temp
gc.collect()

feature_name ="user_nan_count"
df[feature_name] = df["param_1is_nan"] + df["param_2is_nan"] + df["param_3is_nan"] + df["descriptionis_nan"] + df["priceis_nan"]
features.append(feature_name)

############################################ target encoding 
prior = train["deal_probability"].mean()
def add_noise(series, noise_level):
    return series * (1 + noise_level * np.random.randn(len(series)))
def target_encoding(f, smoothing=10, min_samples_leaf=100, noise_level=0.01):
    global features
    global df 
    feature_name = "_".join(f) + "_dl"
    keys = f + ["deal_probability"]
    temp=train[keys].groupby(f)["deal_probability"].agg(["mean", "count"]).reset_index()
    smoothing = 1 / (1 + np.exp(-(temp["count"] - min_samples_leaf) / smoothing))
    temp[feature_name] = prior * (1 - smoothing) + temp["mean"] * smoothing
    temp.drop(["mean", "count"], axis=1, inplace=True)
    df = df.merge(temp, how="left", on=f).fillna(prior)
    df[feature_name] = add_noise(df[feature_name], noise_level)
    features.append(feature_name)
    del temp 
target_encoding(["user_id"],min_samples_leaf=100)
target_encoding(["category_name"])
target_encoding(["image_top_1"])
target_encoding(["param_1"],min_samples_leaf=100)
target_encoding(["param_2"],min_samples_leaf=100)
target_encoding(["city","image_top_1"],min_samples_leaf=100)
target_encoding(["city","category_name"],min_samples_leaf=100)
target_encoding(["region","image_top_1"],min_samples_leaf=100)
target_encoding(["region","category_name"],min_samples_leaf=100)
target_encoding(["param_1","image_top_1"],min_samples_leaf=100)
target_encoding(["param_2","image_top_1"],min_samples_leaf=100)
target_encoding(["param_3","image_top_1"],min_samples_leaf=100)
target_encoding(["param_1","category_name"],min_samples_leaf=100)
target_encoding(["param_2","category_name"],min_samples_leaf=100)
target_encoding(["param_3","category_name"],min_samples_leaf=100)

############################### price 

########### (a) 
feature_name = "category_mean_price_diff"
temp = df[["category_name","price"]].groupby(['category_name'])["price"].mean().reset_index().rename(columns={"price": "category_mean_price"})
df = df.merge(temp, how="left", on=["category_name"])
df[feature_name] = df["price"] - df["category_mean_price"]
features.append(feature_name)
del temp
del df["category_mean_price"]
gc.collect()

feature_name = "image_top_1_count_price_ratio"
temp = df[["image_top_1","price"]].groupby(['image_top_1'])["price"].count().reset_index().rename(columns={"price": "image_top_1_count"})
df = df.merge(temp, how="left", on=["image_top_1"])
df[feature_name] = (df["price"]+1) / (df["image_top_1_count"]+1)
features.append(feature_name)
del temp, df["image_top_1_count"]
gc.collect()

############ (b)
ff = [("region","parent_category_name"),("region", "category_name"), ("city","parent_category_name"),
            ("city", "category_name"),("city","image_top_1")]
for f in ff:
    (f1,f2) = f
    feature_name = f1 + "_" + f2 + "_count"
    temp = df[[f1,f2,"price"]].groupby([f1,f2])["price"].count().reset_index().rename(columns={"price": feature_name})
    df = df.merge(temp, how="left", on=[f1,f2])
    del temp 
    feature_name = f1 + "_" + f2 + "_mean_price"
    temp = df[[f1,f2,"price"]].groupby([f1,f2])["price"].mean().reset_index().rename(columns={"price": feature_name})
    df = df.merge(temp, how="left", on=[f1,f2])
    del temp 

    feature_name = f1 + "_" + f2 + "_count_price_ratio"
    df[feature_name] = df["price"] / (df[f1 + "_" + f2 + "_count"]+1)
    features.append(feature_name)

    feature_name = f1 + "_" + f2 + "_mean_price_diff"
    df[feature_name] = df["price"] - df[f1 + "_" + f2 + "_mean_price"]
    features.append(feature_name)

    feature_name = f1 + "_" + f2 + "_mean_price_ratio"
    df[feature_name] = (df["price"]+1) / (df[f1 + "_" + f2 + "_mean_price"]+1)
    features.append(feature_name)
    del df[f1 + "_" + f2 + "_count"] ,df[f1 + "_" + f2 + "_mean_price"]
gc.collect()

########### #(c)
feature_name = "image_top_1_mean_price_diff_7days"
temp = df[["image_top_1","Weekd of Year","price"]].groupby(['image_top_1',"Weekd of Year"])["price"].mean().reset_index().rename(columns={"price": "image_top_1_mean_price_7days"})
df = df.merge(temp, how="left", on=["image_top_1","Weekd of Year"])
df[feature_name] = df["price"] - df["image_top_1_mean_price_7days"]
features.append(feature_name)
del temp, df["image_top_1_mean_price_7days"]
gc.collect()

########### #(d)
feature_name = "image_top_1_price_order"
temp = df[["item_id","image_top_1","price"]].groupby("image_top_1").apply(lambda g: g.sort_values(["price"]))
temp[feature_name] = temp.groupby(level=0).cumcount()+1
df = df.merge(temp[[feature_name, "item_id"]], how="left", on=["item_id"])
features.append(feature_name)
del temp
gc.collect()

feature_name = "category_name_price_order"
temp = df[["item_id","category_name","price"]].groupby("category_name").apply(lambda g: g.sort_values(["price"]))
temp[feature_name] = temp.groupby(level=0).cumcount()+1
df = df.merge(temp[[feature_name, "item_id"]], how="left", on=["item_id"])
features.append(feature_name)
del temp
gc.collect()

feature_name = "image_top_1_price_order_count"
temp = df[["image_top_1","price"]].groupby(['image_top_1'])["price"].count().reset_index().rename(columns={"price": "image_top_1_count"})
df = df.merge(temp, how="left", on=["image_top_1"])
df[feature_name] = df["image_top_1_price_order"] * (1 / (df["image_top_1_count"]+1))
features.append(feature_name)
del temp

feature_name = "category_name_price_order_count"
temp = df[["category_name","price"]].groupby(['category_name'])["price"].count().reset_index().rename(columns={"price": "category_name_count"})
df = df.merge(temp, how="left", on=["category_name"])
df[feature_name] = df["category_name_price_order"] * (1/ (df["category_name_count"]+1))
features.append(feature_name)
del temp

feature_name = "image_top_1_price_order_7days_count"

temp = df[["item_id","image_top_1","Weekd of Year","price"]].groupby(["image_top_1","Weekd of Year"]).apply(lambda g: g.sort_values(["price"]))
temp["image_top_1_price_order_7days"] = temp.groupby(level=0).cumcount()+1
df = df.merge(temp[["image_top_1_price_order_7days", "item_id"]], how="left", on=["item_id"])
del temp
temp = df[["image_top_1","Weekd of Year","price"]].groupby(["Weekd of Year",'image_top_1'])["price"].count().reset_index().rename(columns={"price": "image_top_1_count_7days"})
df = df.merge(temp, how="left", on=["image_top_1","Weekd of Year"])
del temp

df[feature_name] = df["image_top_1_price_order_7days"] * (1/ (df["image_top_1_count_7days"]+1))
features.append(feature_name)
gc.collect()
############## (e)
region_features = ["image_top_1","category_name"]
class_features = ["param_1","param_2", "param_3"]
for f1 in region_features:
    for f2 in class_features:
        feature_name = f1 + "_" + f2 + "_count"
        temp = df[[f1,f2,"price"]].groupby([f1,f2])["price"].count().reset_index().rename(columns={"price": feature_name})
        df = df.merge(temp, how="left", on=[f1,f2])
        del temp 
        feature_name = f1 + "_" + f2 + "_mean_price"
        temp = df[[f1,f2,"price"]].groupby([f1,f2])["price"].mean().reset_index().rename(columns={"price": feature_name})
        df = df.merge(temp, how="left", on=[f1,f2])
        del temp 
        
        feature_name = f1 + "_" + f2 + "_count_price_ratio"
        df[feature_name] = (df["price"]+1) / (df[f1 + "_" + f2 + "_count"]+1)
        features.append(feature_name)
        
        feature_name = f1 + "_" + f2 + "_mean_price_diff"
        df[feature_name] = df["price"] - df[f1 + "_" + f2 + "_mean_price"]
        features.append(feature_name)
        
        feature_name = f1 + "_" + f2 + "_mean_price_ratio"
        df[feature_name] = (df["price"]+1) / (df[f1 + "_" + f2 + "_mean_price"]+1)
        features.append(feature_name)
        del df[f1 + "_" + f2 + "_count"] ,df[f1 + "_" + f2 + "_mean_price"]
gc.collect()      


# In[7]:


x = df[features].isnull().sum()>0
x[x==True]


# In[8]:


######## feature transform
for f in features:
    if f!="price":
        if df[f].min() >= 0:   df[f] = np.log1p(df[f])
        else:   df[f]= np.log1p(df[f]+16)
    else:
        pass

train = df[:train.shape[0]]
test = df[train.shape[0]:]

f_size = (train[categorical_features].max() - train[categorical_features].min() + 1).values
feature_embed_config = dict(zip(categorical_features,list(zip(f_size, [10]*len(f_size)))))


# In[9]:


len(features)


# In[10]:


train["deal_probability"].isnull().sum()
max_words = embeding_matrix.shape[0]


# In[11]:


x = test[features].isnull().sum()>0
x[x==True]


# In[12]:


from scipy.sparse import vstack, load_npz
def load_imfeatures(folder):
    features = load_npz(folder)    
    return features

ftrain = load_imfeatures('./train-image-features.npz')
ftest = load_imfeatures('./test-image-features.npz')


# In[13]:


ftrain.shape, ftest.shape, train.shape, test.shape


# In[ ]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,mean_squared_error
import lightgbm as lgb    
import math
from keras.layers import *
from keras.models import *
from keras.optimizers import Adam
from keras.callbacks import Callback, TensorBoard

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tf.Session(config=config))

from keras import backend as K
def rmes(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))


algo = "gru-fm-vgg-v2"
units=128
batch_size = 256
epochs = 100
early_stopping_rounds = 3
model_filepath = "./model/{}.hdf5"
local_cv_score = 0 
def get_model():
    inp = Input(shape=(maxlen,)) #maxlen
    hw_inp = Input(shape=(len(features),))
    cf_inp = Input(shape=(len(categorical_features),))
    img_inp = Input(shape=(512,))
    
    ########### text features training 
    x = Embedding(max_words, embed_size, weights=[embeding_matrix], trainable = False)(inp) # (batch_size, maxlen, embed_size)
    x = Bidirectional(CuDNNGRU(units,return_sequences=True))(x)
    max_pool = GlobalMaxPooling1D()(x)
    ave_pool = GlobalAveragePooling1D()(x)
    pool = Concatenate()([max_pool, ave_pool])
    pool = Dropout(0.3)(pool)
    
    ########## image features form vgg
    img_layer = Dense(256)(img_inp)
    img_layer = BatchNormalization()(img_layer)
    img_layer = Activation("relu")(img_layer)
    img_layer = Dropout(0.2)(img_layer)
    
    ############ categorical features embeding
    embeds = []
    for index,cf in enumerate(categorical_features):
        x = Lambda(lambda x: x[:, index, None])(cf_inp)
        x = Embedding(feature_embed_config[cf][0], feature_embed_config[cf][1], trainable=True)(x)
        embeds.append(x) 

    cat_embed = Concatenate(axis=1)(embeds) # (batch_size, len(cat), 10)
    embed = Flatten()(cat_embed)
    embed_fc = Dense(128)(embed)
    embed_fc = BatchNormalization()(embed_fc)
    embed_fc = Activation("relu")(embed_fc)
    embed_fc = Dropout(0.3)(embed_fc)
#     embed_fc = Dense(128)(embed)
#     embed_fc = BatchNormalization()(embed_fc)
#     embed_fc = Activation("relu")(embed_fc)
#     embed_fc = Dropout(0.3)(embed_fc)
    
    ########## numnical features
    dlayer = Dense(128)(hw_inp)
    dlayer = BatchNormalization()(dlayer)
    dlayer = Activation("relu")(dlayer)
    dlayer = Dropout(0.2)(dlayer)
#     dlayer = Dense(128)(hw_inp)
#     dlayer = BatchNormalization()(dlayer)
#     dlayer = Activation("relu")(dlayer)
#     dlayer = Dropout(0.2)(dlayer)
    
    ######### FM part
    t_hw_inp = Reshape((len(features),1))(hw_inp) # hw_inp (batch_size, len(features)) -> (batch_size, len(features),1)
    num_embed = TimeDistributed(Dense(10))(t_hw_inp) # (batch_size, len(features), 10)
    num_embed = Lambda(lambda t: tf.unstack(t, num=len(features),axis=1))(num_embed) #(batch_size, 10)
    cat_embed = [Reshape((10,))(e) for e in embeds]
    factors = cat_embed + num_embed
    s = Add()(factors)
    diffs = [Subtract()([s, x]) for x in factors]
    dots = [Dot(axes=1)([d, x]) for d,x in zip(diffs, factors)]
    fm = Concatenate()(dots)
    fm = BatchNormalization()(fm)    

    pool = Concatenate()([pool,embed_fc,dlayer, img_layer])
    z = Dense(128)(pool)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    z = Dropout(0.2)(z)
    
    pool = Concatenate()([z,fm])
    oup = Dense(1, activation='sigmoid',W_regularizer=None)(pool)
    
    model = Model(input=[inp,hw_inp,cf_inp,img_inp], output=oup)
    model.compile(loss=rmes,optimizer = Adam(lr = 1e-3, decay = 0.0), metrics=['accuracy'])
    return model

class Evaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.min_score = 1
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        global local_cv_score
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            y_pred[y_pred>1] = 1
            y_pred[y_pred<0] = 0
            score = math.sqrt(mean_squared_error(self.y_val, y_pred))
            print("\n RMSE - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score < self.min_score):
                print("*** New LOW Score (previous: %.6f) \n" % self.min_score)
                self.model.save_weights(model_filepath.format(algo))
                self.min_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > early_stopping_rounds:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.min_score))
                    self.model.stop_training = True
                    local_cv_score += self.min_score 
        
        
class TB(TensorBoard):
    def __init__(self, log_every=5, **kwargs):
        super().__init__(**kwargs)
        self.log_every = log_every
        self.counter = 0
    
    def on_batch_end(self, batch, logs=None):
        self.counter+=1
        if self.counter%self.log_every==0:
            for name, value in logs.items():
                if name in ['batch', 'size']:
                    continue
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.writer.add_summary(summary, self.counter)
            self.writer.flush()
        
        super().on_batch_end(batch, logs)

def train_and_evaluate_model(x_tra, y_tra, x_val, y_val):
    model = get_model()
    RMSE = Evaluation(validation_data=(x_val, y_val), interval=1)
    #board=TB(log_dir='./logs', write_graph=False)
    history = model.fit(x_tra, y_tra, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val),
                 callbacks=[RMSE], verbose=1)
    return model

nfolds = 5
skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')

y_pred = np.zeros((test.shape[0],1))
hold_out_preds = np.zeros((train.shape[0],1))

train["psudo_label"] = train["deal_probability"] > 0

x1,x2,x3,x4 = train_words, train[features].values, train[categorical_features].values, ftrain
y = train["deal_probability"].values

z1,z2,z3,z4 = test_words, test[features].values, test[categorical_features].values, ftest

for i, (tra, val) in enumerate(skf.split(train, train["psudo_label"])):
    print ("Running Fold", i+1, "/", nfolds)
    model = train_and_evaluate_model([x1[tra],x2[tra],x3[tra],x4[tra]], y[tra], [x1[val],x2[val],x3[val],x4[val]], y[val])
    model.load_weights(model_filepath.format(algo))
    y_pred += model.predict([z1,z2,z3,z4], batch_size=1024)
    hold_out_preds[val] = model.predict([x1[val],x2[val],x3[val],x4[val]], batch_size=1024)
    
y_pred /= float(nfolds)
y_pred[y_pred>1] = 1
y_pred[y_pred<0] = 0
print("local_cv_score is: ", local_cv_score/nfolds)
hold_out_preds = pd.DataFrame(hold_out_preds)
hold_out_preds.to_csv("./csv02/{}_oofs.csv".format(algo))

submission["deal_probability"] = y_pred
submission.to_csv('./csv02/{}.csv'.format(algo), index=False)

