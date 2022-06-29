#!/usr/bin/env python
# coding: utf-8

# # WEB CRAWLING

# Web crawling merupakan salah satu teknik  dalam mengumpulkan suatu data yang digunakan untuk mengindekskan suatu informasi yang ada pada halaman menggunakan URL (Uniform Resource Locator) disertai dengan API (Application Programming Interface) dalam melakukan penambangan dataset dengan jumlah yang besar. 

# Adapun library yang memiliki fungsi untuk membantu proses crawling data yang ada di webstie yakni Scrapy. Berikut ini merupakan cara install library Scrapy. Adapun proses di bawah ini merupakan proses install dari library scrapy.

# In[1]:


pip install scrapy


# # Import Library

# Setelah berhasil menginstall Scrapy, dapat dilakukan proses crawling data yang ada pada webstie. Pertama dengan mengimportkan library Scrapy yang telah diinstall.

# In[1]:


import scrapy
from scrapy.crawler import CrawlerProcess


# Setelah mengimportkan beberapa library yang akan digunakan dalam proses crawling. Selanjutnya buat sebuah class, dalam class tersebut berisikan nama dari website yang akan dicrawling disini yakni website "Sindonews" dengan keyword "teknologi" dan ditambahkan urlnya. Tidak lupa mensetting format dari file yang akan tersimpan untuk formatnya yakni .csv 
# Tentukan atribut yang akan di crawl yakni judul, tanggal, kategori, dan deskripsi.

# In[2]:


class SpiderWeb(scrapy.Spider):
    name = "sindonews"
    keyword = 'teknologi'
    start_urls = [
        'https://tekno.sindonews.com/'+keyword
        ]
    custom_settings = {
        'FEED_FORMAT': 'csv',
        'FEED_URI': 'berita3.csv'
        }
    
    def parse(self, response):
        for data in response.css('div.desc-news'):
            yield {
                'judul': data.css('div.title a::text').get(),
                'tanggal': data.css('span.date::text').get(),
                'kategori': data.css('span.subkanal::text').get(),
                'deskripsi': data.css('div.desc::text').get()
                }
proses = CrawlerProcess()
proses.crawl(SpiderWeb)
proses.start()


# Crawling data telah berhasil dilakukan, dengan file yang telah tersimpan dengan nama berita3.csv

# # Pre-Processing

# Dalam proses selanjutnya yakni proses Pre-Processing, dalam proses ini memiliki fungsi untuk memastikan kualitas dari data agar data yang digunakan saat analisis data memiliki hasil yang baik dan memiliki data yang clean.

# # Import Library

# Adapun juga library yang dibutuhkan dalam proses pre-processing seperti nltk,swifter,Sastrawi dan masih banyak library yang digunakan. Library-librart tersebut diimport. Seperti dapat dilihat seperti berikut ini 

# In[3]:


import pandas as pd
import numpy as np
import string
import re #regrex libray
import nltk
import swifter
import Sastrawi

from nltk.tokenize import word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.probability import FreqDist
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer


# # Load Dataset

# Proses berikutnya yakni membuat dan menload dataset.Dalam dataset ini dibuat dengan nama "data_berita" kemudian tambahkan file hasil crawling dengan nama file berikut ini "berita3.csv" 

# In[4]:


data_berita = pd.read_csv('berita3.csv')


# Lalu, tampilkan datasetnya dengan .head() data terata akan ditampilkan

# In[5]:


data_berita.head()


# # Case Folding

# Case folding adalag proses pre-processing yang memiliki fungsi untuk menyeragamkan karakter pada data yang ada. Proses Case Folding merupakan proses dalam mengubah huruf kecil. Pada karakter "A-Z" menjadi "a-z"

# In[6]:


# ------ Case Folding --------
# gunakan fungsi Series.str.lower() pada Pandas
data_berita['deskripsi'] = data_berita['deskripsi'].str.lower()


print('Case Folding Result : \n')
print(data_berita['deskripsi'].head(20))
print('\n\n\n')


# Digunakan fungsi series.str.lower() yang terdapat pandas. Atribut yang dilakukan proses case folding yakni "Deskripsi"

# # Tokenizing

# Tokenizing merupakan tahap pemotongan string yang berdasarkan kata-kata yang menyusunkan atau memcahkan kalimat menjadi sebuah kata. 

# Dalam tahap tokenizing yang pertama import string dan juga mengimport library re. Dan juga diperlukan library yang digunakan untuk memproses tokenize yakni nltk.tokenize dan nltk.probability

# In[7]:


import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist


# Adapun proses yang dilakukan yakni :
# 1. remove_tweet_special
# 2. remove_number
# 3. remove_punctuation
# 4. remove_whitespace_LT
# 5. remove_whitespace_multiple
# 6. remove_singl_char
# 7. word_tokenize_wrapper

# In[8]:


def remove_tweet_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
data_berita['deskripsi'] = data_berita['deskripsi'].apply(remove_tweet_special)

#remove number
def remove_number(text):
    return re.sub(r"\d+", "", text)

data_berita['deskripsi'] = data_berita['deskripsi'].apply(remove_number)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

data_berita['deskripsi'] = data_berita['deskripsi'].apply(remove_punctuation)

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

data_berita['deskripsi'] = data_berita['deskripsi'].apply(remove_whitespace_LT)

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

data_berita['deskripsi'] = data_berita['deskripsi'].apply(remove_whitespace_multiple)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

data_berita['deskripsi'] = data_berita['deskripsi'].apply(remove_singl_char)

# NLTK word rokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

data_berita['deskripsi_tokens'] = data_berita['deskripsi'].apply(word_tokenize_wrapper)

print('Tokenizing Result : \n') 
print(data_berita['deskripsi_tokens'].head(20))
print('\n\n\n')


# # Menghitung Frekuensi Distribusi Token

# In[9]:


# NLTK calc frequency distribution
def freqDist_wrapper(text):
    return FreqDist(text)

data_berita['deskripsi_tokens_fdist'] = data_berita['deskripsi_tokens'].apply(freqDist_wrapper)

print('Frequency Tokens : \n') 
print(data_berita['deskripsi_tokens_fdist'].head(20).apply(lambda x : x.most_common()))


# # Filtering (Stopword Removal)

# Stopword removal merupakan tahapan mengambil kata-kata yang penting dari hasil token dengan menggunakan algoritma stoplist (membuang kata yang dinilai kurang penting) atau bisa disebit worldlist (menyimpan kata yang penting). Contoh dari stopword dalam bahasa indonesia yakni sepert "dan","di","dari" dan masih banyak lainnya.

# Gunakan library nltk.cotpus dan import stopwords. Seperti dibawah ini

# Stopword memiliki additional stopword yang dilist. Jadi ada beberapa kata yang tidak diperlukan dalam data. Kata tersebut dimuat dalam list_stopwords.exted
# Kemudian dibuat sebuah fungsi dengan nama stopword_removal yang akan membuang kata yang tidak penting di dalam atribut "deskripsi"

# In[10]:


from nltk.corpus import stopwords
# ----------------------- get stopword from NLTK stopword -------------------------------
# get stopword indonesia
list_stopwords = stopwords.words('indonesian')


# ---------------------------- manualy add stopword  ------------------------------------
# append additional stopword
list_stopwords.extend(["yg", "dg", "rt", "dgn", "ny", "d", 'klo', 
                       'kalo', 'amp', 'biar', 'bikin', 'bilang', 
                       'gak', 'ga', 'krn', 'nya', 'nih', 'sih', 
                       'si', 'tau', 'tdk', 'tuh', 'utk', 'ya', 
                       'jd', 'jgn', 'sdh', 'aja', 'n', 't', 
                       'nyg', 'hehe', 'pen', 'u', 'nan', 'loh', 'rt',
                       '&amp', 'yah'])

# ----------------------- add stopword from txt file ------------------------------------
# read txt stopword using pandas
txt_stopword = pd.read_csv("berita3.csv", names= ["stopwords"], header = None)

# convert stopword string to list & append additional stopword
list_stopwords.extend(txt_stopword["stopwords"][0].split(' '))

# ---------------------------------------------------------------------------------------

# convert list to dictionary
list_stopwords = set(list_stopwords)


#remove stopword pada list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

data_berita['deskripsi_tokens_WSW'] = data_berita['deskripsi_tokens'].apply(stopwords_removal) 


print(data_berita['deskripsi_tokens_WSW'].head(20))


# # Normalization

# Normalization merupakan salah satu tahapan yang ada dalam pre-processing. Normalization memiliki fungsi untuk menyamakan nilai yang ada dalam data dalam rentan tertentu.

# In[11]:


normalizad_word = pd.read_csv("berita3.csv")

normalizad_word_dict = {}

for index, row in normalizad_word.iterrows():
    if row[0] not in normalizad_word_dict:
        normalizad_word_dict[row[0]] = row[1] 

def normalized_term(document):
    return [normalizad_word_dict[term] if term in normalizad_word_dict else term for term in document]

data_berita['deskripsi_normalized'] = data_berita['deskripsi_tokens_WSW'].apply(normalized_term)

data_berita['deskripsi_normalized'].head(20)


# # Stemming 

# Stemming juga merupakan salah satu tahap yang ada di dalam teks pre-processing. Stemming sendiri memiliki fungsi guna mengubah term ke dalam bentuk akar. Stemming bisasny menghilangkan kata imbuhan seperti awalan maupun akhiran.

# In[12]:



# import Sastrawi package
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in data_berita['deskripsi_normalized']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

data_berita['deskripsi_tokens_stemmed'] = data_berita['deskripsi_normalized'].swifter.apply(get_stemmed_term)
print(data_berita['deskripsi_tokens_stemmed'])


# # Simpan Hasil Pre-Processing ke CSV

# Setalah tahapan pre-processing telah dilakukan, maka data dari hasil pre-processing tersebut disimpan ke dalam format .csv
# Seperti berikut ini 

# In[13]:


data_berita.to_csv("Text_Preprocessing_Berita1.csv")


# # Modeling Using LSA

# Tahap selanjutnya yakni membuat model dengan menggunakan LSA atau Latent Semantic Analysis

# # Import Library

# Dalam tahap modeling menggunakan LSA juga diperlukan library tersendiri gun mendukung proses modeling. Adapun library yang diperlukan yakni numpy, pandas, matplotlib, seaborn, dan nltk. 
# Berikut ini merupakan proses import yang dilakukan

# In[14]:


# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#import nltk
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize,sent_tokenize

#preprocessing
from nltk.corpus import stopwords  #stopwords
from nltk import word_tokenize,sent_tokenize # tokenizing
from nltk.stem import PorterStemmer,LancasterStemmer  # using the Porter Stemmer and Lancaster Stemmer and others
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer  # lammatizer from WordNet

# for named entity recognition (NER)
from nltk import ne_chunk

# vectorizers for creating the document-term-matrix (DTM)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

#stop-words
stop_words=set(nltk.corpus.stopwords.words('english'))


# Dalam proses import tersebut ada library yang digunakan dalam visualisasi dan manipulasi data diantaranya yakni :
# 1. numpy
# 2. pandas
# 3. matplotlib
# 4. seaborn
# 
# Serta library yang digunakan dalam pre-processing yakni :
# 1. nltk

# # Load Dataset

# Dalam proses modeling juga dilakukan pembuatan dataframe disini data framenya dinamai seperti yang sebelumnya yakni "data_berita". Dan data yang digunakan merupakan data hasil preprocessing dengan nama file "Text_Preprocessing_Berita.csv"

# In[15]:


data_berita = pd.read_csv('berita3.csv')


# Lalu, tampilkan datasetnya dengan .head() data terata akan ditampilkan

# In[16]:


data_berita.head()


# Sebelum beralih pada tahap selanjutnya, ada beberapa atribut yang perlu dihapus yakni judul, tanggal, dan kategori. Dikarenakan atribut yang digunakan yakni atribut deskripsi.

# In[17]:


# drop judul, tanggal, dan kategori
data_berita.drop(['judul','tanggal','kategori'],axis=1,inplace=True)


# Tampilkan data yang telah di hapus atributnya dan menyisakan deskripsi

# In[18]:


data_berita.head()


# # Data Cleaning & Pre-Processing

# Pada tahap selnajutnya yakni melakukan cleaning dan juga tahap pre-processing kembali. Menggunakan lemmatizer danjuga stemmer. Terdapat juga kata-kata yang tidak digubakan bersama dan juga kata-kata yang panjanya kurang dari 3 karakter yang berguna untuk mengurangi beberapa kata yang tidak cocok.

# In[19]:


def clean_text(headline):
  le=WordNetLemmatizer()
  word_tokens=word_tokenize(headline)
  tokens=[le.lemmatize(w) for w in word_tokens if w not in stop_words and len(w)>3]
  cleaned_text=" ".join(tokens)
  return cleaned_text


# In[20]:


# time taking
data_berita['deskripsi_cleaned_text']=data_berita['deskripsi'].apply(clean_text)


# In[21]:


data_berita.head()


# Dapat dilihat perbedaan setelah dilakukan stopwords dan juga kata yang lebih pendek. Juga ada beberapa kata yang telah diestimasi seperti "calls"-->"call"

# Lalu hapus kolom yang belum dilakuak pre-processing, disini adalah kolom deskripsi

# In[22]:


data_berita.drop(['deskripsi'],axis=1,inplace=True)


# In[23]:


data_berita.head()


# Dapat dilihat bahwa ada beberapa judul berita tertentu

# In[24]:


data_berita['deskripsi_cleaned_text'][0]


# # Extracting Feature dan Membuat Dokumen Term-Matrik (DTM)

# DTM memiliki nilai TF-Idf
# Tentukan pula parameter dari vectore TF-Idf

# Adapun beberapa poin penting yang perlu diperhatoakn :
# 1. LSA pada umumnua diimplementasikan dengan menggunakan nilai TF-Idff dan tidak dengan Count Vectorizer
# 2. max_feature bergantubg pada daya komputasi dan juga pada eval. Metrik merupakan skor yang koheren untuk menentukan model
# 3. Nilai default untuk min_df dan max_df dapat bekerja dengan baik
# 4. Dapat mencoba nilai yang berbeda-beda dalam ngram_range

# In[25]:


vect =TfidfVectorizer(stop_words=stop_words, max_features=1000) # to play with. min_df,max_df,max_features etc...


# In[26]:


vect_text=vect.fit_transform(data_berita['deskripsi_cleaned_text'])


# Dalam hasilnya, dapat dilihat kata yang sering muncul dan kata yang jarang muncul dalam berita yang ada dalam idf. Apabila memiliki nilai yang kecil maka katanya lebih umum digunakan dalam berita utama

# In[27]:


print(vect_text.shape)
print(vect_text)


# In[29]:


idf=vect.idf_


# In[30]:


dd=dict(zip(vect.get_feature_names(), idf))
l=sorted(dd, key=(dd).get)
# print(l)
print(l[0],l[-1])
print(dd['yang'])
print(dd['york'])  # police is most common and forecast is least common among the news headlines.


# Oleh karena itu, berdasarkan pada nilai idf yang ada 'adil' adalah kata yang paling sering muncul sedangkan 'alat' paling jarang muncul dalam berita

# # Topik Modeling

# # Latent Semantic Analysis (LSA)

# Pada pendekatan pertama digunakan LSA. LSA pada dasarnya adalah dekomposisi dari nilai tunggal.

# SVD akan menguraikan DTM menajdi tiga matriks = S=U.(sigma).(V.T). Nilai matrik U menunjukkan matriks dari dokumen topik sementara (V) adalah matriks dari term.

# Pada setiap baris dari matriks U (matriks istilah dari dokumen) merupakan representasi vektor yang ada dalam dokumen yang sesuai. Panjang vektor ini ialah jumlah topik yang diinginkan. Representasi dari vektor untuk suku yang ada dalam data dapat ditemui dalam matriks V.

# Jadi, SVD memberikan nilai vektor pada setiap dokumen dan juga istilah dalam data. Panjang dari setiap vektor adalah k. Vektor ini digunakan untuk menentukan kata dan dokumen serupa dalam metode kesamaan kosinus. 

# Dapat digunakan fungsi truncastedSVD untuk mengimplementasikan LSA. Parameter n_components merupakan jumlah topik yang akan diekstrak. Model tersebut nantinya akan di fit dan ditransformasikan pada hasil yang diberikan oleh vectorizer.

# Tahap terakhir yakni LSA dan LSI (I digunakan untuk mengindekskan) ialah sama dan yang terakhir digunakan dalam konteks pencarian sebuah informasi.

# In[30]:


from sklearn.decomposition import TruncatedSVD
lsa_model = TruncatedSVD(n_components=10, algorithm='randomized', n_iter=10, random_state=42)

lsa_top=lsa_model.fit_transform(vect_text)


# In[31]:


print(lsa_top)
print(lsa_top.shape)  # (no_of_doc*no_of_topics)


# In[32]:


l=lsa_top[0]
print("Document 0 :")
for i,topic in enumerate(l):
  print("Topic ",i," : ",topic*100)


# Hampir sama dengan dokumen lainnya, dapat dilakukan proses seperti di bawah ini. Akan tetapi perlu diperhatikan bahwa dalam setiap nilai tidak menambah 1 itu bukan sebuah kemungkinan topik yang ada dalam dokumen.

# In[33]:


print(lsa_model.components_.shape) # (no_of_topics*no_of_words)
print(lsa_model.components_)


# Sehingga, didapatkan sebuah list dari kata-kata yang penting dan memiliki makna dari setiap 10 topic yang ditampilkan. Sederhananya dibawah ini ditampilkan 10 kata dalam setiap topic.

# In[34]:


# most important words for each topic
vocab = vect.get_feature_names()

for i, comp in enumerate(lsa_model.components_):
    vocab_comp = zip(vocab, comp)
    sorted_words = sorted(vocab_comp, key= lambda x:x[1], reverse=True)[:10]
    print("Topic "+str(i)+": ")
    for t in sorted_words:
        print(t[0],end=" ")
    print("\n")


# In[ ]:




