
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import gc

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""


# In[2]:


train = pd.read_csv("train.csv",parse_dates=["activation_date"])
del train["image"]
test = pd.read_csv("test.csv",parse_dates=["activation_date"])
# train_zero = train[train["deal_probability"]<=0].sample(frac=0.2)
# train_nonzero = train[train["deal_probability"]>0]
# train = pd.concat([train_zero, train_nonzero])

y_psudo_labels = train["deal_probability"] > 0
ytrain = train["deal_probability"].values

aggregated_features = pd.read_csv("aggregated_features.csv")


# In[3]:


#def feature_processing(df):
categorical_features_tobe = [ "region", "city", "parent_category_name", "category_name" ,"user_type","param_1","param_2","param_3","image_top_1"] 
features = ["price", "item_seq_number"] 
categorical_features = []
df = pd.concat([train,test], axis=0)

#############filling NA
df["price"] = df["price"].fillna(99999999)
df["image_top_1"] = df["image_top_1"].fillna(df["image_top_1"].max()+1)
df["param_1"] = df["param_1"].fillna("missing")
df["param_2"] = df["param_2"].fillna("missing")
df["param_3"] = df["param_3"].fillna("missing")
df["description"] = df["description"].fillna("something")

df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["Day of Month"] = df['activation_date'].dt.day
categorical_features.extend(["Weekday","Weekd of Year","Day of Month"])
    
##########label encode
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
for cf in categorical_features_tobe:
    le.fit(df[cf].astype(str))
    le_feature_name = "{}_le".format(cf)
    df[le_feature_name] = le.transform(df[cf].astype(str))
    categorical_features.append(le_feature_name)
del le 
gc.collect()

###########add target encoding (mean)
for cf in categorical_features_tobe[1:]:
    new_f = "{}_dl".format(cf)
    temp = train[[cf,"deal_probability"]].groupby(cf).mean().reset_index().rename(columns={"deal_probability": new_f})
    df = df.merge(temp, how="left", on=cf)
    df[new_f] = np.log1p(df[new_f])
    df[new_f] = df[new_f].fillna(df[new_f].mean())
    features.append(new_f)
    del temp
gc.collect()

###### text features 
from sklearn.feature_extraction.text import TfidfVectorizer as Tfidf
from scipy.sparse import hstack, csr_matrix

tfidf = Tfidf(ngram_range=(1,2), max_features=20000, sublinear_tf=True)
textfeats = ["description", "title"]
for cols in textfeats:
    df[cols] = df[cols].astype(str) 
    df[cols] = df[cols].astype(str).fillna('missing') # FILL NA
    df[cols] = df[cols].str.lower() # Lowercase all text, so that capitalized words dont get treated differently
    df[cols + '_num_words'] = df[cols].apply(lambda comment: len(comment.split())) # Count number of Words
    df[cols + '_num_unique_words'] = df[cols].apply(lambda comment: len(set(w for w in comment.split())))
    df[cols + '_words_vs_unique'] = df[cols+'_num_unique_words'] / df[cols+'_num_words'] * 100 # Count Unique Word
    features.extend([cols + '_num_words',cols + '_num_unique_words', cols + '_words_vs_unique'])
    
df["text"] = df["title"].astype(str) + " " + df["description"].astype(str)
x = tfidf.fit_transform(df["text"]) 
features.extend(categorical_features)

######### add aggragated features
df = df.merge(aggregated_features, how="left", on="user_id")
features.extend(aggregated_features.columns[1:])
df[aggregated_features.columns[1:]] = df[aggregated_features.columns[1:]].fillna(df[aggregated_features.columns[1:]].mean())

######## feature transform
df["item_seq_number"]=np.log1p(df["item_seq_number"])
df["price"] = np.log1p(df["price"])
df[aggregated_features.columns[1:]] = np.log1p(df[aggregated_features.columns[1:]])

train = hstack((csr_matrix(df[:train.shape[0]][features].values),x[:train.shape[0]]))
test = hstack((csr_matrix(df[train.shape[0]:][features].values),x[train.shape[0]:]))
features += tfidf.get_feature_names()

del x
del df
gc.collect()


# In[5]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,mean_squared_error
import lightgbm as lgb    
import math
def train_model(params, x_train, y_train, objective='regression', metrics='rmse',
                 feval=None, num_boost_round=200, verbose_eval=100):
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.02,
        #'is_unbalance': 'true',  #because training data is unbalance (replaced with scale_pos_weight)
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.8,  # Subsample ratio of the training instance.
        'subsample_freq': 0,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.3,  # Subsample ratio of columns when constructing each tree.
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'reg_alpha': 0,  # L1 regularization term on weights
        'reg_lambda': 0,  # L2 regularization term on weights
        'nthread': 8,
        'verbose': 0,
        'metric':metrics
    }
    lgb_params.update(params)
    xgtrain = lgb.Dataset(x_train, label=y_train,
                          feature_name=features,
                          categorical_feature=categorical_features
                          )
    evals_results = {}
    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain], 
                     valid_names=['train'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=None,
                     verbose_eval=verbose_eval, 
                     feval=feval)

    f, ax = plt.subplots(figsize=[10,10])
    lgb.plot_importance(bst, max_num_features=50, ax=ax)
    return bst

params = {
    'learning_rate': 0.1,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 200,  # 2^max_depth - 1
    'max_depth': -1,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'subsample': 0.8,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 20,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight': 1, # because training data is extremely unbalanced 
    'reg_lambda ' : 0.001,   
}

xtrain = train.tocsr()
xtest = test.tocsr()

model = train_model(params, xtrain, ytrain)
model.save_model("./model/lgb_alldata")


# In[6]:


import lightgbm as lgb 
model = lgb.Booster(model_file="./model/lgb_alldata")


# In[7]:


train_features = model.predict(xtrain, pred_leaf=True)
test_features = model.predict(xtest, pred_leaf=True)


# In[8]:


train_features.shape, test_features.shape


# In[9]:


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

algo = "lgb_feed_nn_v1"
units=128
batch_size = 256
epochs = 100
early_stopping_rounds = 5
model_filepath = "./model/{}.hdf5"
local_cv_score = 0 
def get_model():
    inp = Input(shape=(200,)) 
    embeds = []
    for index,cf in enumerate(categorical_features):
        x = Lambda(lambda x: x[:, index, None])(inp)
        x = Embedding(200, 1, trainable=True)(x)
        embeds.append(x) 
    embed = Concatenate(axis=1)(embeds) 
    embed = Flatten()(embed)
    embed = Dropout(0.3)(embed)

    embed_fc = Dense(256)(embed)
    embed_fc = BatchNormalization()(embed_fc)
    embed_fc = Activation("relu")(embed_fc)
    
    z = Dense(256)(embed_fc)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    z = Dropout(0.2)(z)
    
    z = Dense(64)(z)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    z = Dropout(0.2)(z)
    
    z = Dense(64)(z)
    z = BatchNormalization()(z)
    z = Activation("relu")(z)
    z = Dropout(0.2)(z)
    
    oup = Dense(1, activation='sigmoid',W_regularizer=None)(z)
    
    model = Model(input=inp, output=oup)
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


x1 = train_features
y = ytrain

z1 = test_features

for i, (tra, val) in enumerate(skf.split(train, y_psudo_labels)):
    print ("Running Fold", i+1, "/", nfolds)
    model = train_and_evaluate_model(x1[tra], y[tra], x1[val], y[val])
    model.load_weights(model_filepath.format(algo))
    y_pred += model.predict(z1, batch_size=1024)
    hold_out_preds[val] = model.predict(x1[val], batch_size=1024)
    
y_pred /= float(nfolds)
y_pred[y_pred>1] = 1
y_pred[y_pred<0] = 0
print("local_cv_score is: ", local_cv_score/nfolds)
hold_out_preds = pd.DataFrame(hold_out_preds)
hold_out_preds.to_csv("./csv/{}_oofs.csv".format(algo))

submission["deal_probability"] = y_pred
submission.to_csv('./csv/{}.csv'.format(algo), index=False)

