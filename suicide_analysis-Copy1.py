#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data=pd.read_csv("Suicides in India 2001-2012.csv")
data


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


data.describe()


# In[6]:


data.isnull().sum()


# In[7]:


data.head()


# In[8]:


data.tail()


# In[9]:


data.columns


# In[10]:


data.sample(10)


# # Exploratory Data Analysis

# In[11]:


sns.heatmap(data.corr(),annot=True)
plt.show()


# In[12]:


data.hist()


# In[13]:


sns.countplot(data.Year)


# In[14]:


sns.countplot(data.Type_code)


# In[15]:


sns.countplot(data.Gender)


# In[16]:


sns.countplot(data.Age_group)


# In[17]:


plt.figure(figsize=(15,5))
sns.distplot(np.log1p(data.Year), hist=False, color="b", kde_kws={"shade": True})
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Distribution of Gender")
plt.show()


# In[18]:


plt.figure(figsize=(15,5))
sns.distplot(np.log1p(data.Total), hist=False, color="b", kde_kws={"shade": True})
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Distribution of Gender")
plt.show()


# In[19]:


fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(data.Year, data.Total, errwidth=0)
plt.ylabel('Year Suicide Cases')
plt.title('Year wise suicide attempts', fontdict = {'fontsize' : 15})


# In[20]:


fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(data.Gender, data.Year, errwidth=0)
plt.ylabel('Gender')
plt.title('Gender wise suicide attempts', fontdict = {'fontsize' : 15})


# In[21]:


fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(data.Year, data.Type_code, errwidth=0)
plt.ylabel('Type_code')
plt.title('type of wise suicide attempts', fontdict = {'fontsize' : 15})


# In[22]:


def label_encoding(categories):
    categories = list(set(list(categories.values)))
    mapping = {}
    for idx in range(len(categories)):
        mapping[categories[idx]] = idx
    return mapping


# In[23]:


data['Gender'] = data['Gender'].map(label_encoding(data['Gender']))
data.head(10)


# In[24]:


plt.figure(figsize=(15,5))
sns.distplot(np.log1p(data.Total), hist=False, color="b", kde_kws={"shade": True})
plt.xlabel("Gender")
plt.ylabel("Frequency")
plt.title("Distribution of Gender")
plt.show()


# In[25]:


data['Type_code'] = data['Type_code'].map(label_encoding(data['Type_code']))
data.head(10)


# In[26]:


predict_data = data[['Year', 'Total','Type_code']].values
targe_data= data[['Gender']].values


# In[27]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(predict_data,targe_data,test_size=0.20,random_state=0)


# In[28]:


#random forest
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

max_accuracy = 0

for x in range(5):
    rf = RandomForestClassifier(random_state=x)
    rf.fit(X_train,Y_train)
    Y_pred_rf = rf.predict(X_test)
    current_accuracy = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
c    if(current_accuracy>max_accuracy):
        max_accuracy = current_accuracy
        best_x = x

rf = RandomForestClassifier(random_state=best_x)
rf.fit(X_train,Y_train)
Y_pred = rf.predict(X_test)


# In[ ]:


Y_pred


# In[ ]:


#print("\n",rf.predict(X_test))
print("\nAccuracy Random Forest : "+ str(round(accuracy_score(rf.predict(X_test), Y_test[0:])*100, 1)))


# In[ ]:


"""
#svm
from sklearn import svm

sv = svm.SVC(kernel='linear')

sv.fit(X_train,Y_train)

Y_pred_svm = sv.predict(X_test)
"""


# In[ ]:


#score_svm = round(accuracy_score(Y_pred_svm,Y_test)*100,2)

#print("The accuracy score achieved using Linear SVM is: "+str(score_svm)+" %")


# In[ ]:


#LogisticRegression
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

lr.fit(X_train,Y_train)

Y_pred = lr.predict(X_test)


# In[ ]:


acc_lr = round(accuracy_score(Y_pred,Y_test)*100,2)

print("The accuracy score achieved using Logistic Regression is: "+str(acc_lr))


# # Sentiment Analysis

# In[ ]:


import re
import csv


# In[ ]:


data = pd.read_csv('Twitter_Suicide_Data_new.csv')
data


# In[ ]:


data.columns


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


#text preprocessing
def processTweet2(tweet):

    tweet = tweet.lower()
    tweet.lower()
    tweet = re.sub('((www\.[^\s]+)|(https?://[^\s]+))','URL',tweet)
    tweet = re.sub('@[^\s]+','AT_USER',tweet)
    tweet = re.sub('[\s]+', ' ', tweet)
    tweet = re.sub('[\n]+', ' ', tweet)
    tweet = re.sub(r'[^\w]', ' ', tweet)
    tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
    tweet = tweet.replace(':)','')
    tweet = tweet.replace(':(','')
    tweet = tweet.strip('\'"')
    re.sub('[^A-Za-z0-9]+', '', tweet)
    replaceTwoOrMore(tweet)
    return tweet


# In[ ]:


def getStopWordList(stopWordListFileName):
    stopWords = []
    stopWords.append('at_user')
    stopWords.append('url')

    fp = open(stopWordListFileName, 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopWords.append(word)
        line = fp.readline()
    fp.close()
    return stopWords



# In[ ]:


def replaceTwoOrMore(s):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", s)


# In[ ]:


stopWords = []
st = open('stopwords.txt', 'r')
stopWords = getStopWordList('stopwords.txt')
stopWords


# In[ ]:


for i in range(1,303):
    tweet=data.iloc[i,1]
    data.iloc[i,1]=processTweet2(tweet)


# In[ ]:


def getFeatureVector(tweet):
    features = []

    words = tweet.split()
    for w in words:

        w = replaceTwoOrMore(w)
        w = w.strip('0123456789')
        w = w.strip('\'"!?,.')
        if (w == ""):
            continue
        elif(w in stopWords):
            continue
        else:
            features.append(w.lower())

    return features


# In[ ]:


for i in range(0,303):
    tweet=data.iloc[i,1]
    a=getFeatureVector(tweet)
    data.iloc[i,1] = " ".join(a)
    print(a)    


# In[ ]:


tweets = []
featureList = []
for i in range(0,303):
    sentiment = data['Sentiment'][i]
    tweet = data['Content'][i]
    processedTweet = processTweet2(tweet)
    featureVector = getFeatureVector(processedTweet)
    featureList.extend(featureVector)
    tweets.append((featureVector, sentiment))


# In[ ]:


def extract_features1(tweet):
    processedTweet = processTweet2(tweet)
    featureVector = getFeatureVector(processedTweet)
    print(featureVector)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in featureVector)
    return features


# In[ ]:


def extract_features(tweet):
    tweet_words = set(tweet)
    print(tweet_words)
    features = {}
    for word in featureList:
        features['contains(%s)' % word] = (word in tweet_words)
    return features


# In[ ]:


featureList


# In[ ]:


fe=extract_features("hate myself")
fe


# In[ ]:


import nltk
train_set = nltk.classify.util.apply_features(extract_features, tweets)
NBClassifier = nltk.NaiveBayesClassifier.train(train_set)


# In[ ]:


train_set


# In[ ]:


NBClassifier.show_most_informative_features()


# In[ ]:


NBClassifier.classify(extract_features("I dont have hope"))


# In[ ]:


NBClassifier.classify(extract_features("I can't"))


# In[ ]:


NBClassifier.classify(extract_features1("I love india"))


# In[ ]:


NBClassifier.classify(extract_features1("I love myself"))


# In[ ]:


sorted(NBClassifier.labels())


# In[ ]:


NBClassifier.prob_classify(extract_features("I like you"))


# In[ ]:


extract_features("love")


# In[ ]:


test=[extract_features1("sorry"),extract_features1("i can do anythong")]
for pdist in NBClassifier.prob_classify_many(test):
    print('%.4f %.4f' % (pdist.prob('Positive'), pdist.prob('Negative')))


# In[ ]:


extract_features1("be happy")


# In[ ]:


NBClassifier.classify(extract_features1("i love chocolate"))


# In[ ]:


data.Sentiment.value_counts()


# In[ ]:


data['Sentiment_num'] = data.Sentiment.map({'Negative':0, 'Positive':1})


# In[ ]:


data.head(30)


# In[ ]:


X = data.Content
y = data.Sentiment_num


# In[ ]:


print(X.shape)
print(y.shape)


# In[ ]:


#split the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
print("\n",X_train.shape)
print("\n",X_test.shape)
print("\n",y_train.shape)
print("\n",y_test.shape)


# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
vect.fit(X_train)
X_train_dtm = vect.transform(X_train)
X_train_dtm


# In[ ]:


X_test_dtm = vect.transform(X_test)
X_test_dtm


# In[ ]:


#naive bayes algorithm
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[ ]:


nb.fit(X_train_dtm, y_train)
class_y_pred = nb.predict(X_test_dtm)


# In[ ]:


from sklearn import metrics
print(metrics.accuracy_score(y_test, class_y_pred))
print(metrics.confusion_matrix(y_test, class_y_pred))


# In[ ]:


prob_y_pred= nb.predict_proba(X_test_dtm)[:, 1]
prob_y_pred


# In[ ]:


metrics.roc_auc_score(y_test, prob_y_pred)


# In[ ]:





# In[ ]:





# In[ ]:




