
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import gc

train = pd.read_csv("train.csv",parse_dates=["activation_date"])
test = pd.read_csv("test.csv",parse_dates=["activation_date"])

y_psudo_labels = train["deal_probability"] > 0
ytrain = train["deal_probability"].values

aggregated_features = pd.read_csv("aggregated_features.csv")
lda_features = pd.read_csv("lda_features.csv")


# In[2]:


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
# df["image_top_1"] = df["image_top_1"].fillna(df["image_top_1"].max()+1)
# df["param_1"] = df["param_1"].fillna("missing")
# df["param_2"] = df["param_2"].fillna("missing")
# df["param_3"] = df["param_3"].fillna("missing")
# # temp = df[["category_name", "price"]].groupby("category_name")["price"].median().reset_index().rename(columns={"price": "category_median_price"})
# # df = df.merge(temp, how="left", on="category_name")
# # df["price"] = df["price"].fillna(df["category_median_price"])
# df["price"] = df["price"].fillna(99999999)

# fs = ["param_1", "param_2", "param_3", "image_top_1", "price"]
# train[fs] = df[fs][:train.shape[0]]

# df["price"] = np.log1p(df["price"])

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

############################### weekday 

df["Weekday"] = df['activation_date'].dt.weekday
df["Weekd of Year"] = df['activation_date'].dt.week
df["day"] = df['activation_date'].dt.dayofyear
categorical_features.extend(["Weekday",])

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

feature_name ="user_mean_description_length"
temp = df[["user_id","description_num_words"]].groupby(['user_id'])["description_num_words"].mean().reset_index().rename(columns={"description_num_words": feature_name})
df = df.merge(temp, how="left", on=["user_id"])
features.append(feature_name)
del temp
gc.collect()

feature_name ="user_nan_count"
df[feature_name] = df["param_1is_nan"] + df["param_2is_nan"] + df["param_3is_nan"] + df["descriptionis_nan"] + df["priceis_nan"]
features.append(feature_name)

###################################### lda features
df = df.merge(lda_features, how="left", on="item_id")
features.extend(lda_features.columns[:-1])

features.extend(categorical_features)


# In[3]:


target_features_list = []
count_features_list = []
price_features_list = []
new_price_features_list = []

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
    df = df.merge(temp, how="left", on=f)
    df[feature_name] = df[feature_name].fillna(prior)
    df[feature_name] = add_noise(df[feature_name], noise_level)
    features.append(feature_name)
    target_features_list.append(feature_name)
    del temp 
def count_features(f):
    global df
    global features
    feature_name = "_".join(f)+"_count"
    group_keys = f + ["item_id"]
    temp = df[group_keys].groupby(f)["item_id"].count().reset_index().rename(columns={"item_id":feature_name})
    df = df.merge(temp, on=f, how="left")
    features.append(feature_name)
    count_features_list.append(feature_name)
    del temp
    gc.collect()
def mean_price_features(f):
    global df
    global features
    feature_name = "_".join(f)+"_mean_price"
    group_keys = f + ["price"]
    temp = df[group_keys].groupby(f)["price"].mean().reset_index().rename(columns={"price":feature_name})
    df = df.merge(temp, on=f, how="left")
    features.append(feature_name)
    price_features_list.append(feature_name)
    del temp
    gc.collect()
def price_statics_features(f):
    global df
    global features
    f = "_".join(f)
    feature_name = f + "_count_price_ratio"
    df[feature_name] = (df["price"]+1) / (df[f+"_count"]+1)
    features.append(feature_name)
    price_features_list.append(feature_name)
    feature_name = f + "_mean_price_diff"
    df[feature_name] = df["price"] - df[f+"_mean_price"]
    features.append(feature_name)
    price_features_list.append(feature_name)

def unique_features(f):
    global df
    global features
    feature_name = "_".join(f) + "_nunique"
    temp = df[f].groupby(f[:-1])[f[-1]].nunique().reset_index().rename(columns={f[-1]:feature_name})
    df = df.merge(temp, on=f[:-1], how="left")
    features.append(feature_name)
    del temp
    gc.collect()
    
def time_delta_features(f):
    global df
    global features
    feature_name = "_".join(f) + "_next_day_delta"
    temp = df[f+["day","item_id"]].groupby(f).apply(lambda g:g.sort_values(["day"]))
    temp[feature_name] = temp["day"].transform(lambda x: x.diff().shift(-1))
    df = df.merge(temp[["item_id",feature_name]],how="left",on="item_id")
    features.append(feature_name)
    del temp
    gc.collect()
    
# unique_features(["user_id","image_top_1"])
# unique_features(["user_id","category_name"])

# unique_features(["image_top_1","param_1"])
# unique_features(["image_top_1","param_2"])
# unique_features(["image_top_1","param_3"])

# unique_features(["city","image_top_1"])

# unique_features(["image_top_1","param_1","param_2","param_3","city"])
# unique_features(["image_top_1","param_1","param_2","param_3","user_id"])
# unique_features(["image_top_1","param_1","param_2","param_3","day"])
# unique_features(["category_name","param_1","param_2","param_3","city"])
# unique_features(["category_name","param_1","param_2","param_3","user_id"])
# unique_features(["category_name","param_1","param_2","param_3","day"])


# In[4]:


# def quantile_price_features(f):
#     global df
#     global features
#     feature_name = "_".join(f)+"_quantile10_price"
#     group_keys = f + ["price"]
#     temp = df[group_keys].groupby(f)["price"].quantile(0.1).reset_index().rename(columns={"price":feature_name})
#     df = df.merge(temp, on=f, how="left")
#     features.append(feature_name)
#     new_price_features_list.append(feature_name)
#     del temp
#     feature_name = "_".join(f)+"_quantile25_price"
#     group_keys = f + ["price"]
#     temp = df[group_keys].groupby(f)["price"].quantile(0.25).reset_index().rename(columns={"price":feature_name})
#     df = df.merge(temp, on=f, how="left")
#     features.append(feature_name)
#     new_price_features_list.append(feature_name)
#     del temp
#     feature_name = "_".join(f)+"_median_price"
#     group_keys = f + ["price"]
#     temp = df[group_keys].groupby(f)["price"].median().reset_index().rename(columns={"price":feature_name})
#     df = df.merge(temp, on=f, how="left")
#     features.append(feature_name)
#     new_price_features_list.append(feature_name)
#     del temp
#     feature_name = "_".join(f)+"_quantile10_price_diff"
#     df[feature_name] = df["price"] - df["_".join(f)+"_quantile10_price"]
#     features.append(feature_name)
#     new_price_features_list.append(feature_name)
#     feature_name = "_".join(f)+"_quantile25_price_diff"
#     df[feature_name] = df["price"] - df["_".join(f)+"_quantile25_price"]
#     features.append(feature_name)
#     new_price_features_list.append(feature_name)
#     feature_name = "_".join(f)+"_median_price_diff"
#     df[feature_name] = df['price'] - df["_".join(f)+"_median_price"]
#     features.append(feature_name)
#     new_price_features_list.append(feature_name)
#     gc.collect()

# fff=[["category_name"],["image_top_1"],["region","category_name"],["region","image_top_1"],["city","category_name"],
#     ["city","image_top_1"],["category_name","param_1"],["image_top_1","param_1"],["region","category_name","param_1"],
#      ["region","image_top_1","param_1"],["city","category_name","param_1"],["city","image_top_1","param_1"]]
# for f in fff:
#     quantile_price_features(f)
# df[new_price_features_list + ["item_id"]].to_csv("quantile_price_features_df.csv", index=False)


# In[5]:


# order_features_list = []
# def order_features(f):
#     global df
#     global features
#     feature_name = "_".join(f)+"_order"
#     temp = df[f+["item_id"]].groupby(f[:-1]).apply(lambda g: g.sort_values(f[-1]))
#     temp[feature_name] = temp.groupby(level=0).cumcount()+1
#     df = df.merge(temp[[feature_name, "item_id"]], how="left", on=["item_id"])
#     features.append(feature_name)
#     order_features_list.append(feature_name)
#     del temp
#     gc.collect()
# fff=[["category_name","price"],["image_top_1","price"],["region","category_name","price"],["region","image_top_1","price"],["city","category_name","price"],
#     ["city","image_top_1","price"],["category_name","param_1","price"],["image_top_1","param_1","price"],["region","category_name","param_1","price"],
#      ["region","image_top_1","param_1","price"],["city","category_name","param_1","price"],["city","image_top_1","param_1","price"]]
# for f in fff:
#     order_features(f)
# df[order_features_list + ["item_id"]].to_csv("order_features_df.csv", index=False)    


# In[78]:


# from sklearn.decomposition import TruncatedSVD as SVD
# def get_features_latent_vector(f, dim=5):
#     global df
#     global features
#     x = df[[f, "image_top_1", "deal_probability"]].groupby([f,"image_top_1"])["deal_probability"].mean()
#     x = x.unstack().fillna(0).astype(np.float32)
#     y = SVD(n_components=dim).fit_transform(x.values)
#     temp = pd.DataFrame(dict(zip(list(x.index),y))).T
#     temp[f] = temp.index
#     cols = []
#     for i in range(dim):
#         feature_name =  f+"_latent_vector_%d"%(i)
#         cols.append(feature_name)
#     features.extend(cols)
#     cols.append(f)
#     temp.columns = cols
#     df= df.merge(temp, on=f, how="left")
#     del x,y,cols,temp
#     gc.collect()
# get_features_latent_vector("region",3)
# get_features_latent_vector("city",3)
# get_features_latent_vector("param_1",3)
# get_features_latent_vector("param_2",3)
# get_features_latent_vector("param_3",3)


# In[6]:


count_features_df = pd.read_csv("./count_features_df.csv")
df = df.merge(count_features_df, how="left", on="item_id")
features.extend(count_features_df.columns[:-1])

price_features_df = pd.read_csv("./price_features_df.csv")
df = df.merge(price_features_df, how="left", on="item_id")
features.extend(price_features_df.columns[:-1])

new_price_features_df = pd.read_csv("./quantile_price_features_df.csv")
df = df.merge(new_price_features_df, how="left", on="item_id")
features.extend(new_price_features_df.columns[:-1])

order_features_df = pd.read_csv("./order_features_df.csv")
df = df.merge(order_features_df, how="left", on="item_id")
features.extend(order_features_df.columns[:-1])

target_features = pd.read_csv("./target_features_df.csv")
df = df.merge(target_features, how="left", on="item_id")
features.extend(target_features.columns[:-1])

### filling NAN after computing all
count_features_list = count_features_df.columns[:-1]
price_features_list = price_features_df.columns[:-1]
order_features_list = order_features_df.columns[:-1]
new_price_features_list = new_price_features_df.columns[:-1]

df[count_features_list]= df[count_features_list].fillna(1)
for f in price_features_list:
    df[f]=df[f].fillna(df[f].max())
for f in order_features_list:
    df[f]=df[f].fillna(df[f].max())
for f in new_price_features_list:
    df[f]=df[f].fillna(df[f].max())
for f in features:
    df[f]=df[f].fillna(df[f].max())


# In[7]:


train_image_feature = pd.read_csv("./train_blurrness.csv")
test_image_feature = pd.read_csv("./test_blurrness.csv")
image_feature = pd.concat([train_image_feature, test_image_feature])
df = df.merge(image_feature, on="item_id", how="left")
features.append(image_feature.columns[1])
del train_image_feature, test_image_feature, image_feature
gc.collect()


# In[9]:


img_feature = pd.read_csv("./image_features.csv")
df = df.merge(img_feature, on="item_id", how="left")
features.extend(img_feature.columns[1:])
del img_feature
gc.collect()


# In[6]:


# train_nn_pred = pd.read_csv("./csv02/bilstm-fm-v2_oofs.csv")
# test_nn_pred = pd.read_csv("./csv02/bilstm-fm-v2.csv")
# nn_pred = pd.concat([train_nn_pred,test_nn_pred], axis=0)
# df["nn_pred"] = nn_pred["0"].values
# features.append("nn_pred")
# df["nn_pred"]= 0
# test_nn_pred["nn_pred"] = test_nn_pred["deal_probability"]
# df = df.merge(test_nn_pred[["item_id","nn_pred"]], on="item_id", how="left")
# df["nn_pred"][:train.shape[0]] = train_nn_pred["0"].values


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
from nltk.corpus import stopwords 

russian_stop = set(stopwords.words('russian'))
tfidf_para = {
    "stop_words": russian_stop,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    "smooth_idf":False
}
tfidf = TfidfVectorizer(ngram_range=(1,3), max_features=20000, sublinear_tf=True)
df["text"] = df["title"].astype(str) + " " + df["description"].astype(str)
tf_df = tfidf.fit_transform(df["text"]) 
del df["text"]
gc.collect()


# In[13]:


df["im_tsvd_30"].isnull().sum()


# In[14]:


# train_ = df[:train.shape[0]]
# test_ = df[train.shape[0]:]
# features.extend(categorical_features)
# features = list(set(features))
try:
    del train_, test_
except:
    pass
# gc.collect()
# features.extend(categorical_features)
# features = features[:-len(tfidf.get_feature_names())]
train_ = hstack((csr_matrix(df[:train.shape[0]][features].values),tf_df[:train.shape[0]]))
test_ = hstack((csr_matrix(df[train.shape[0]:][features].values),tf_df[train.shape[0]:]))
features += tfidf.get_feature_names()


# In[15]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,mean_squared_error
import lightgbm as lgb
import math

algo= "lgb_featureEngineer_img_v1"

def train_and_evaluate_model(params, x_train, y_train, x_val, y_val, objective='regression', metrics='rmse',
                 feval=None, early_stopping_rounds=50, num_boost_round=10000, verbose_eval=100):
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
        'nthread': 16,
        'verbose': 0,
        'metric':metrics
    }
    lgb_params.update(params)
    print("preparing validation datasets")
    xgtrain = lgb.Dataset(x_train, label=y_train,
                          feature_name=features,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(x_val, label=y_val,
                          feature_name=features,
                          categorical_feature=categorical_features
                          )
    evals_results = {}
    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval, 
                     feval=feval)

    print("\nModel Report")
    print("bst.best_iteration: ", bst.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst.best_iteration-1])

    hold_out_preds = bst.predict(x_val, num_iteration= bst.best_iteration)
    score = math.sqrt(mean_squared_error(y_val, hold_out_preds))
    print("rmse score: ", score)
    f, ax = plt.subplots(figsize=[10,30])
    lgb.plot_importance(bst, max_num_features=100, ax=ax)
    return bst,bst.best_iteration, score

params = {
    'learning_rate': 0.015,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 200,  # 2^max_depth - 1
    'max_depth': 8,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'subsample': 0.8,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'reg_lambda ' : 0,   
}

nfolds = 5
skf = StratifiedKFold(n_splits=nfolds, shuffle=True)
submission = pd.read_csv('sample_submission.csv')

y_pred = np.zeros((test.shape[0],))
hold_out_preds = np.zeros((train.shape[0],)) 
 
xtrain = train_.tocsr() #train_[features]#train_.tocsr()   
xtest = test_.tocsr() #test_[features]#test_.tocsr()

local_cv_score = 0    
for i, (tra, val) in enumerate(skf.split(train, y_psudo_labels)): # this is used for fix the split 
    print ("Running Fold", i+1, "/", nfolds)
    model, best_iter, score = train_and_evaluate_model(params, xtrain[tra], ytrain[tra], xtrain[val], ytrain[val])
    y_pred += model.predict(xtest, num_iteration=best_iter)
    hold_out_preds[val] = model.predict(xtrain[val], num_iteration=best_iter)
    local_cv_score += score
y_pred /= float(nfolds)
y_pred[y_pred>1] = 1
y_pred[y_pred<0] = 0
print("local_cv_score is: ", local_cv_score/nfolds)
hold_out_preds = pd.DataFrame(hold_out_preds)
hold_out_preds.to_csv("./csv02/{}_oofs.csv".format(algo))

submission["deal_probability"] = y_pred
submission.to_csv('./csv02/{}.csv'.format(algo), index=False)


# In[26]:


from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss,mean_squared_error
import lightgbm as lgb    
import math

def train_and_evaluate_model(params, x_train, y_train, x_val, y_val, features, categorical_features, objective='regression', metrics='rmse',
                 feval=None, early_stopping_rounds=50, num_boost_round=5000, verbose_eval=100):
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
        'nthread': 16,
        'verbose': 0,
        'metric':metrics
    }
    lgb_params.update(params)
    print("preparing validation datasets")
    xgtrain = lgb.Dataset(x_train, label=y_train,
                          feature_name=features,
                          categorical_feature=categorical_features
                          )
    xgvalid = lgb.Dataset(x_val, label=y_val,
                          feature_name=features,
                          categorical_feature=categorical_features
                          )
    evals_results = {}
    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets=[xgtrain, xgvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=verbose_eval, 
                     feval=feval)

    print("\nModel Report")
    print("bst.best_iteration: ", bst.best_iteration)
    print(metrics+":", evals_results['valid'][metrics][bst.best_iteration-1])

    hold_out_preds = bst.predict(x_val, num_iteration= bst.best_iteration)
    score = math.sqrt(mean_squared_error(y_val, hold_out_preds))
    print("rmse score: ", score)
#     f, ax = plt.subplots(figsize=[10,30])
#     lgb.plot_importance(bst, max_num_features=100, ax=ax)
    return bst,bst.best_iteration, score

params = {
    'learning_rate': 0.02,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 200,  # 2^max_depth - 1
    'max_depth': 8,  # -1 means no limit
    'min_child_samples': 100,  # Minimum number of data need in a child(min_data_in_leaf)
    'subsample': 0.8,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'reg_lambda ' : 0,
}

nfolds = 5
skf = StratifiedKFold(n_splits=nfolds, shuffle=True)


def feature_sampling_lgb(sample_features,rounds):
    algo= "lgb_fe_sample_v2_5fold_v%d"%(rounds)
    local_cv_score = 0 
    sample_categorical_features = categorical_features
    sample_features.extend(categorical_features)
    sample_features = list(set(sample_features))
    
    train_ = hstack((csr_matrix(df[:train.shape[0]][sample_features].values),tf_df[:train.shape[0]]))
    test_ = hstack((csr_matrix(df[train.shape[0]:][sample_features].values),tf_df[train.shape[0]:]))
    sample_features += tfidf.get_feature_names()

    submission = pd.read_csv('sample_submission.csv')
    y_pred = np.zeros((test.shape[0],))
    hold_out_preds = np.zeros((train.shape[0],)) 

    xtrain = train_.tocsr()   
    xtest = test_.tocsr()
    
    local_cv_score = 0    
    for i, (tra, val) in enumerate(skf.split(train, y_psudo_labels)): # this is used for fix the split 
        print ("Running Fold", i+1, "/", nfolds)
        model, best_iter, score = train_and_evaluate_model(params, xtrain[tra], ytrain[tra], xtrain[val], ytrain[val], sample_features, sample_categorical_features)
        y_pred += model.predict(xtest, num_iteration=best_iter)
        hold_out_preds[val] = model.predict(xtrain[val], num_iteration=best_iter)
        local_cv_score += score
    y_pred /= float(nfolds)
    y_pred[y_pred>1] = 1
    y_pred[y_pred<0] = 0
    print("local_cv_score is: ", local_cv_score/nfolds)
    hold_out_preds = pd.DataFrame(hold_out_preds)
    hold_out_preds.to_csv("./csv03/{}_oofs.csv".format(algo))

    submission["deal_probability"] = y_pred
    submission.to_csv('./csv03/{}.csv'.format(algo), index=False)
    
    del train_, test_, xtrain, xtest, hold_out_preds, y_pred,submission
    gc.collect()
    return local_cv_score 

import random
for i in range(10):
    print("Sampling Features at %d Rounds"%(i))
    sample_features = random.sample(features, 150)
    cv = feature_sampling_lgb(sample_features,i)


# In[83]:


[x for x in categorical_features if features.index(x)>-1]


# In[48]:


x = sorted(list(zip(model.feature_name(), model.feature_importance(importance_type="gain"))), key = lambda x:x[1], reverse=True)


# In[5]:



# ############################################ target encoding 
# ff = ["city","region","user_type","parent_category_name","category_name","image_top_1","param_1","param_2","param_3"]
# for i1,f1 in enumerate(ff):
#     target_encoding([f1])
#     for i2,f2 in enumerate(ff):
#         if i1>=i2: continue
#         target_encoding([f1,f2])
# f1 = ["city","region"]
# f2 = ["image_top_1","category_name","parent_category_name"]
# f3 = ["user_type","param_1","param_2","param_3"]
# fff = [("param_1","param_2","param_3"),]
# for i1 in f1:
#     for i2 in f2:
#         for i3 in f3:
#             fff.append((i1,i2,i3))
# for f in fff:
#     (f1,f2,f3) = f
#     target_encoding([f1,f2,f3])
# target_encoding(["image_top_1","param_1","param_2","param_3"])
# target_encoding(['category_name',"param_1","param_2", "param_3"])

# ######################################## count features
# ff = ["user_id","city","region","user_type","parent_category_name","category_name","image_top_1","param_1","param_2","param_3"]
# for i1,f1 in enumerate(ff):
#     count_features([f1])
#     for i2,f2 in enumerate(ff):
#         if i1>=i2: continue
#         count_features([f1,f2])
# count_features(['category_name',"param_1","param_2", "param_3"])
# count_features(['image_top_1',"param_1","param_2", "param_3"])

# f1 = ["city","region"]
# f2 = ["image_top_1","category_name","parent_category_name"]
# f3 = ["user_type","param_1","param_2","param_3"]
# fff = [("param_1","param_2","param_3"),]
# for i1 in f1:
#     for i2 in f2:
#         for i3 in f3:
#             fff.append((i1,i2,i3))
# for f in fff:
#     count_features(list(f))

# count_features(["day"])
# count_features(["day","user_id"])
# count_features(["day","user_id","category_name"])
# count_features(["day","city"])
# count_features(["day","image_top_1"])
# count_features(["day","category_name"])
# count_features(["day","image_top_1","user_type"])
# count_features(["day","city","image_top_1"])
# count_features(["day","city","category_name"])
# count_features(["day","region","image_top_1"])
# count_features(["day","region","category_name"])
# count_features(["day","city","image_top_1","user_type"])
# count_features(["day","city","image_top_1","param_1"])
# count_features(["day","city","image_top_1","param_2"])
# count_features(["day","city","image_top_1","param_3"])
# count_features(["day","city","image_top_1","param_1","param_2","param_3"])

# count_features(["user_id",'category_name',"param_1","param_2", "param_3"])
# count_features(["user_id",'image_top_1',"param_1","param_2", "param_3"])

# ####################################### mean price features 
# ff = ["user_id","city","region","user_type","parent_category_name","category_name","image_top_1","param_1","param_2","param_3"]
# for i1,f1 in enumerate(ff):
#     mean_price_features([f1])
#     for i2,f2 in enumerate(ff):
#         if i1>=i2: continue
#         mean_price_features([f1,f2])
# f1 = ["city","region"]
# f2 = ["image_top_1","category_name","parent_category_name"]
# f3 = ["user_type","param_1","param_2","param_3"]
# fff = [("param_1","param_2","param_3"),]
# for i1 in f1:
#     for i2 in f2:
#         for i3 in f3:
#             fff.append((i1,i2,i3))
# for f in fff:
#     mean_price_features(list(f))

# mean_price_features(['category_name',"param_1","param_2", "param_3"])
# mean_price_features(['image_top_1',"param_1","param_2", "param_3"])

# mean_price_features(["day"])
# mean_price_features(["day","user_id"])
# mean_price_features(["day","user_id","category_name"])
# mean_price_features(["day","city"])
# mean_price_features(["day","image_top_1"])
# mean_price_features(["day","category_name"])
# mean_price_features(["day","image_top_1","user_type"])
# mean_price_features(["day","city","image_top_1"])
# mean_price_features(["day","city","category_name"])
# mean_price_features(["day","region","image_top_1"])
# mean_price_features(["day","region","category_name"])
# mean_price_features(["day","city","image_top_1","user_type"])
# mean_price_features(["day","city","image_top_1","param_1"])
# mean_price_features(["day","city","image_top_1","param_2"])
# mean_price_features(["day","city","image_top_1","param_3"])
# mean_price_features(["day","city","image_top_1","param_1","param_2","param_3"])

# ########################################price statics features
# ff = ["user_id","city","region","user_type","parent_category_name","category_name","image_top_1","param_1","param_2","param_3"]
# for i1,f1 in enumerate(ff):
#     price_statics_features([f1])
#     for i2,f2 in enumerate(ff):
#         if i1>=i2: continue
#         price_statics_features([f1,f2])

# f1 = ["city","region"]
# f2 = ["image_top_1","category_name","parent_category_name"]
# f3 = ["user_type","param_1","param_2","param_3"]
# fff = [("param_1","param_2","param_3"),]
# for i1 in f1:
#     for i2 in f2:
#         for i3 in f3:
#             fff.append((i1,i2,i3))
# for f in fff:
#     price_statics_features(list(f))
# price_statics_features(['category_name',"param_1","param_2", "param_3"])
# price_statics_features(['image_top_1',"param_1","param_2", "param_3"])

# price_statics_features(["day"])
# price_statics_features(["day","city"])
# price_statics_features(["day","image_top_1"])
# price_statics_features(["day","category_name"])
# price_statics_features(["day","image_top_1","user_type"])
# price_statics_features(["day","city","image_top_1"])
# price_statics_features(["day","city","category_name"])
# price_statics_features(["day","region","image_top_1"])
# price_statics_features(["day","region","category_name"])
# price_statics_features(["day","city","image_top_1","user_type"])
# price_statics_features(["day","city","image_top_1","param_1"])
# price_statics_features(["day","city","image_top_1","param_2"])
# price_statics_features(["day","city","image_top_1","param_3"])
# price_statics_features(["day","city","image_top_1","param_1","param_2","param_3"])

# count_features(["city","image_top_1","param_1","param_2","param_3"])
# mean_price_features(["city","image_top_1","param_1","param_2","param_3"])
# price_statics_features(["city","image_top_1","param_1","param_2","param_3"])


# In[4]:


# def price_delta_features(f):
#     global df
#     global features
#     feature_name = "_".join(f) + "_price_delta"
#     temp = df[f+["price","item_id"]].groupby(f).apply(lambda g:g.sort_values(["day"]))
#     temp[feature_name] = temp["day"].transform(lambda x: x.diff().shift(-1))
#     df = df.merge(temp[["item_id",feature_name]],how="left",on="item_id")
#     features.append(feature_name)
#     del temp
#     gc.collect()
# time_delta_features(["category_name"])
# time_delta_features(["city","category_name"])
# time_delta_features(["region","category_name"])

# time_delta_features(["image_top_1"])
# time_delta_features(["city","image_top_1"])
# time_delta_features(["region","image_top_1"])

# time_delta_features(["image_top_1","param_1","param_2","param_3"])
# time_delta_features(["city","image_top_1","param_1","param_2","param_3"])
# time_delta_features(["region","image_top_1","param_1","param_2","param_3"])

# time_delta_features(["user_id"])
# time_delta_features(["user_id","image_top_1"])
# time_delta_features(["user_id","category_name"])
# time_delta_features(["user_id","image_top_1","param_1","param_2","param_3"])

