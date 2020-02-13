#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Algoritmo desarrollado por Adriana Cárdenas como parte del trabajo de fin de máster de la visualización masiva de datos - BigData - UNIR
#Feb-13-2020
#Este algoritmo permite consultar cualquier tema de interés de un plan de desarrollo territorial y genera un rerpote
#con información estadística y representaciones visuales con el resultado de las búsqueda.

#!pip install nltk #Librería para procesamiento de lenguaje natural
#nltk.download()
#!pip install chardet #Se usa para detectar el encode de los archivos de texto
#!pip install tika #Se usa para leer archivos PDF
#!pip install -U spacy
#!pip install -U spacy
#python -m spacy download en_core_web_sm
#!pip install gensim
#!pip install pyLDAvis
#!pip install wordcloud
#!pip install reportlab
#!pip install selenium phantomjs


# In[2]:


from __future__ import division
import nltk #La librería para procesamiento de lenguaje natural
#nltk.download()
import chardet #Se usa para detectar el encode de los archivos planos
import codecs #Se usa para convertir los archivos planos a codificación UTF-8
import re #Permite usar expresiones regulares en el tratamiento de cadenas de texto
import string
import csv
from pprint import pprint
import pandas as pd #Se utiliza para realizar cálculos estadísticos
import os
from nltk.probability import FreqDist
from nltk.tokenize import sent_tokenize #Se usa para tokenizar el texto en un vector de sentencias
from nltk.tokenize import word_tokenize #Se usa para tokenizar un vector en palabras
from tika import parser #Función para pasar de PDF a TXT
from nltk.corpus import stopwords #Librería que contiene el listado de stop words
stopword = stopwords.words('spanish')
import spacy
model_spacy = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
import gensim
#nltk.download('omw')
from nltk.corpus import wordnet
import gensim.corpora as corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel

import pyLDAvis
import pyLDAvis.gensim  
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors
from collections import Counter
from matplotlib.ticker import FuncFormatter
import  numpy as np
import time


from sklearn.manifold import TSNE
from bokeh.plotting import figure, output_file, show
from bokeh.models import Label
from bokeh.io import output_notebook
from bokeh.io import export_png
#conda install -c conda-forge phantomjs

from reportlab.pdfgen import canvas
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import *


# In[3]:


#Estandarizar a UTF8 la codificación de caracteres de los archivos planos. Al realizar las pruebas 
#de carga se identificó que Python no reconocía algunos caracteres por problemas en la codificación

def conv_utf8(nomarch):
    linea = []
    infile1 = ruta1 + nomarch
    infile2 = ruta2 + nomarch
    rawdata = open(infile1, 'rb').read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    sourceFile = codecs.open(infile1, "r", charenc)
    targetFile = codecs.open(infile2, "w", "utf-8")
    contents = sourceFile.read()
    targetFile.write(contents)


# In[4]:


#Leer un archivo TXT para guardarlo en un dataframe, cada línea del archivo será una fila 
#con el fin de conservar la estructura de los párrafos.

def txt_dataf(nomarch):
    infile1 = ruta2 + nomarch
    f = open(infile1, "r", encoding="utf-8")
    texto = []
    for linea in f:
        texto.append(linea)
    tokens = pd.DataFrame({"texto_original": texto}) 
    return tokens


# In[5]:


#Leer el dataframe generado por la función txt_data con el fin de eliminar cualquier de puntuación, 
#dobles espacios o saltos de página.

def limpieza(tk):
    tk = re.sub('\n','',tk)
    tk = re.sub(' +',' ',tk)
    re_punc = re.compile('[%s]' % re.escape(string.punctuation)) 
    tk = re_punc.sub('', tk)
    return tk


# In[6]:


#ELeer el dataframe generado por la función txt_data con el fin de eliminar cualquier carácter 
#diferente a las letras y dejar todo el texto en minúsculas

def depuracion(tk):
    tk = tk.lower()
    tk = tk.replace('á','a').replace('é','e').replace('í','i').replace('ó','o').replace('ú','u').replace('ü','u')
    tk = re.sub(r"[^a-z|ñ. ]","",tk)
    return tk


# In[7]:


#Adicionar palabras vacías al arreglo que viene por defecto en la librería de 
#NLTK de Python, con el fin de mejorar el resultado da la normalización.

def sword_otrasf():
    global stopword
  
    rawdata = open(r'/Volumes/GoogleDrive/My Drive/TFM/lista_stopWord.txt', 'rb').read()
       
    infile1 = '/Volumes/GoogleDrive/My Drive/TFM/lista_stopWord.txt'
    rawdata = open(infile1, 'rb').read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    sourceFile = codecs.open(infile1, "r", charenc)
    contents = sourceFile.read()
    contents = contents.split(",")
    stopword = stopword + contents


# In[8]:


#Eliminar las palabras vacías

def stop_word(tk):
    tkw = word_tokenize(tk)
    tkw_sw = [w for w in tkw if not w in stopword]
    return tkw_sw


# In[9]:


#Convertir una palabra a su forma básica significativa considerando el contexto. Se utilizó la librería spacy

def lematizar(tk):  
    bigram = gensim.models.Phrases(tk, min_count=5, threshold=100) 
    trigram = gensim.models.Phrases(bigram[tk], threshold=100)  
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    
    tkb1 = [bigram_mod[w] for w in tk]
    tkb2 = [trigram_mod[bigram_mod[w]] for w in tkb1]
    cadena_lem = []
    for x in tk:
        w =  model_spacy(" ".join(x))
        cadena_lem.append([tok.lemma_ for tok in w]) 
    return cadena_lem


# In[10]:


#Leer un archivo plano original y aplicar todas las funciones de limpieza, depuración y normalización 
#con el fin de obtener un archivo óptimo para las tareas de extracción y minería

def preprocesamiento(nomarch):
    conv_utf8(nomarch)
    tokens = txt_dataf(nomarch)
    tokens['texto_limpio'] = tokens['texto_original'].apply(limpieza)
    tokens.head(5)
    tokens['texto_depurado'] = tokens['texto_limpio'].apply(depuracion)
    tokens = tokens[tokens['texto_depurado'].map(len)>0]
    tokens['stop_word'] = tokens['texto_depurado'].apply(stop_word)
    tokens['lemma'] = lematizar(tokens['stop_word'])
    return tokens
    tokens.head(25)


# In[11]:


#Extender el arreglo de palabras vacías con una lista personalizada que se recibe como parámetro. 
#Esto se utiliza cuando al revisar los resultados del modelo se detectan palabras que no aportan significado al contexto

def stop_word_extend(stopw):
    global stopword
    stopw.strip()
    stopw = stopw.split(',')
    stopword = stopword + stopw

    cadena = ""
    for x in stopw:
        w =  model_spacy(str(x)) 
        lm = [tok.lemma_ for tok in w]
        cadena = cadena + str(lm) + ","
        
    cadena = cadena.replace("['","").replace("']","").replace(' ','')
    cadena = cadena.split(',')
    stopword = stopword + (cadena)


# In[12]:


#Lematizar el diccionario de palabras clave de los planes de desarrollo territorial provisto por el DNP

def lematizar_pclave():
    archivo = pd.read_csv (r'/Volumes/GoogleDrive/My Drive/TFM/lista_palabrasDNP.csv',delimiter=';')
    lista = pd.DataFrame(archivo, columns= ['primaria','secundaria'])
    cadena_lem = []
    for x in lista['primaria']:
        w =  model_spacy(str(x)) 
        cadena_lem.append([tok.lemma_ for tok in w]) 
    lista['lemma_primaria'] = cadena_lem

    cadena_lem = []
    for x in lista['secundaria']:
        w =  model_spacy(x) 
        cadena_lem.append([tok.lemma_ for tok in w]) 
    lista['llave'] = cadena_lem
    
    i = 0
    for x in lista['lemma_primaria']: 
        lista['lemma_primaria'][i] = " ".join(x)
        i += 1
    
    i = 0
    for x in lista['llave']: 
        lista['llave'][i] = " ".join(x)  
        i += 1

    lista.to_csv(r'/Volumes/GoogleDrive/My Drive/TFM/lemma_palabrasDNP.csv')


# In[13]:


#Generar el arreglo de palabras de contexto

def palabras_clave(cadena_interes):
    #Lee lista de palabras del DNP
    archivo = pd.read_csv (r'/Volumes/GoogleDrive/My Drive/TFM/lemma_palabrasDNP.csv')
    lista = pd.DataFrame(archivo, columns= ['primaria','secundaria','lemma_primaria','llave'])
    cadena_interes = pd.DataFrame(cadena_interes.split(','), columns = ['llave'])
    cadena_lem = []
    for x in cadena_interes['llave']:
        w =  model_spacy(x) 
        cadena_lem.append([tok.lemma_ for tok in w]) 

    #Genera un dataframe con las palabras de interes lematizadas
    cadena_intlm = pd.DataFrame(cadena_lem, columns = ['llave'])   
    cadena_interes = cadena_interes.append(cadena_intlm,ignore_index=True)
    
    #Cruza las palabras lematizadas del DNP con las palabras de interés lematizadas para encontrar coincidencias
    cruce = pd.merge(lista, cadena_intlm, on='llave')

    if (len(cruce)>0):
        #Genera una lista única de las palabras clave principales o primarias del DNP
        cruce_p = cruce['lemma_primaria']
        cruce_p = cruce_p.drop_duplicates()
        #Extrae todas las palabras secundarias en lenguaje original que tienen relación con las palabras primarias que cruzaron
        cruce_p = pd.merge(lista, cruce_p, on='lemma_primaria')
        cruce_s = pd.DataFrame(cruce_p['secundaria'], columns = ['llave'])
        cruce_l = pd.DataFrame(cruce_p['llave'], columns = ['llave'])
        cadena_interes = cadena_interes.append(cruce_s,ignore_index=True)
        cadena_interes = cadena_interes.append(cruce_l,ignore_index=True)
   
    cadena_interes = cadena_interes.drop_duplicates()

    #Busca los sinonimos en Wordnet de la lista de palabras de contxto
    cadena = " "
    for palabra in cadena_interes['llave']:
        for syn in wordnet.synsets(palabra, lang='spa'):
            for lemma in syn.lemmas('spa'):
                cadena += lemma.name() + " "
    
    contexto_wn = pd.DataFrame(cadena.split(), columns = ['llave']) 
    contexto_wn = contexto_wn.append(cadena_interes,ignore_index=True)
    contexto_wn = contexto_wn.drop_duplicates()
    
    contexto_wn['texto_limpio'] = contexto_wn['llave'].apply(depuracion)
    
    #Lematización de las palabras del contexto
    cadena_lem = []
    for x in contexto_wn['texto_limpio']:
        w =  model_spacy(str(x)) 
        cadena_lem.append([tok.lemma_ for tok in w]) 
    
    contexto_wn_tl = pd.DataFrame(cadena_lem, columns = ['llave'])
  
    contexto_wn_tl = contexto_wn_tl.drop_duplicates()
    return contexto_wn_tl


# In[14]:


#Encontrar el modelo con el número óptimo de clústeres

def compute_coherence_values(dictionary, corpus, texts, limit, start, step):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus,id2word=id2word, num_topics=num_topics)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())      
    return model_list, coherence_values


# In[15]:


#Generar el corpus filtrado con el texto que presente coincidencias con las palabras de contexto

def extraccion(tokens,contexto):
    tokens_f = pd.DataFrame(columns=('texto_original', 'texto_limpio', 'texto_depurado', 'stop_word', 'lemma'))
    for index, row in tokens.iterrows(): 
        df = pd.DataFrame(row['lemma'], columns = ['llave'])
        cr = pd.merge(df, contexto, on = 'llave')
        if (len(cr)>0):
            texto_original = row['texto_original']
            texto_limpio = row['texto_limpio']
            texto_depurado = row['texto_depurado']
            stop_word = row['stop_word']
            lemma = row['lemma']    
            tokens_f.loc[len(tokens_f)] = [texto_original,texto_limpio,texto_depurado,stop_word,lemma] 
    return tokens_f


# In[16]:


#Guardar en una variable de texto la información de los tópicos del modelo seleccionado, 

def display_topics(model):
    cadena = ""
    for topic_idx, topic in enumerate(model.print_topics()):
        nom_topic = " ".join(re.findall( r'\"(.[^"]+).?', topic[1]))
        print ("Topic %d:" % (topic_idx))
        print (" ".join(re.findall( r'\"(.[^"]+).?', topic[1])), "\n")
        cadena += "Tema " + str(topic_idx) + ": <br />\n" + nom_topic + "<br />\n<br />\n"
    return cadena


# In[17]:


#Generar una lista con una etiqueta para cada tema, esta variable se utiliza en el archivo que contiene 
#los textos filtrados y que se puede utilizar en le reporte

def etiqueta_topico(model):
    etiqueta = {}
    for topic_idx, topic in enumerate(model.print_topics()):
        idx = str(topic_idx)
        etiqueta[topic_idx] = "Topico " + idx
    return etiqueta


# In[18]:


#Guardar en un dataframe  la información de los tópicos del modelo seleccionado, esto incluye número 
#del tema, % de contribución del tema y lista de palabras clave. Esta información se utiliza en el reporte de PDF

def format_topics_sentences(ldamodel=0, corpus="", texts=0):
    sent_topics_df = pd.DataFrame()

    # Obtiene el tópico dominanate de cada documento
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Obtiene el tópico dominante, % de contribución y palabra clave de cada tema
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Topico_Dominante', 'Porc_Contribucion', 'PalabrasClave_Topico']

    # Adiciona el tecto original al objeto de salida
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


# In[19]:


#Identificar a qué tópico pertenece cada uno de los párrafos del texto

def mapeo_topico(lda_model,corpus_filtrado):
    corpus_filtrado.head(5)
    df_topic_sents_keywords = format_topics_sentences(lda_model, corpus=corpus, texts=corpus_filtrado['lemma'].to_list())
    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Topico_Dominante', 'Topic_Porc_Contribucion', 'PalabraClave', 'Texto']
    df_dominant_topic['Topico_Dominante'] = df_dominant_topic['Topico_Dominante'].astype('int64')
    # Show
    #df_dominant_topic.head(10)
    etiqueta_t = etiqueta_topico(lda_model)
    df_dominant_topic['Topico_Dominante'] = df_dominant_topic['Topico_Dominante'].map(etiqueta_t)
    df_dominant_topic.head(10)
    corpus_filtrado['topico'] = df_dominant_topic['Topico_Dominante']
    corpus_filtrado[['texto_depurado', 'topico']].head(10)
    return corpus_filtrado


# In[20]:


#Generar un reporte dinámico que permita visualizar la distribución de los clústeres y la frecuencia de 
#las palabras que los conforman

def davis(lda_model, corpus, id2word):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, '/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/lda_model.html')
    #vis


# In[21]:


#Representar las 50 palabras con mayor frecuencia tanto el corpus completo como en el corpus filtrado

def freq_corpus(tokens,archivo,titulo):
    cadena = ""
    fig = plt.figure(figsize=(15,6))
  
    for palabra in tokens:
        for w in palabra:
            cadena += w +","
    cadena =  cadena.split(",")
    ax0 = nltk.FreqDist(cadena)
    ax0.plot(50)
    #plt.show()
    fig.savefig("/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/" + archivo + ".jpg")
    return set(cadena)    


# In[22]:


#Representar la frecuencia de palabras de cada tema y su % de distribución

def frecuencias_palabras(lda_model,ntopic,long_f,weight):
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in corpus_filtrado['lemma'] for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['palabra', 'topic_id', 'importancia', 'conteo_palabra'])        

    # Plot Word Count and Weights of Topic Keywords
    ntopic = ntopic + 2 
    if ntopic == 1: filas = 1;col = 1
    if ntopic == 2: filas = 1;col = 2
    if ntopic == 3: filas = 2;col = 2
    if ntopic == 4: filas = 2;col = 2
    if ntopic == 5: filas = 3;col = 2
    if ntopic == 6: filas = 3;col = 2
    if ntopic == 7: filas = 4;col = 2
    if ntopic == 8: filas = 4;col = 2
    if ntopic == 9: filas = 5;col = 2
    if ntopic == 10: filas = 5;col = 2
    fig, axes = plt.subplots(filas, col, figsize=(10,10), sharey=True, dpi=240)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='palabra', height="conteo_palabra", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.5, label='Conteo Palabra')
        ax_twin = ax.twinx()
        ax_twin.bar(x='palabra', height="importancia", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Conteo Palabra', color=cols[i])
        ax_twin.set_ylim(0, 0.2); ax.set_ylim(0, long_f)
        ax.set_title('Topico: ' + str(i), color=cols[i], fontsize=10)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'palabra'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
        if i == (ntopic-1):
            break

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Frecuencia de palabras clave por tópico', fontsize=12, y=1.05)    
    fig.savefig("/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/wfreq_temas.jpg")


# In[23]:


#Representar la frecuencia de palabras de cada tema y su % de distribución

def frecuencias_palabras2(lda_model,ntopic,long_f,weight):
    topics = lda_model.show_topics(formatted=False)
    data_flat = [w for w_list in corpus_filtrado['lemma'] for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    # Plot Word Count and Weights of Topic Keywords
    ntopic = ntopic + 2 
    if ntopic == 1: filas = 1;col = 1
    if ntopic == 2: filas = 1;col = 2
    if ntopic == 3: filas = 2;col = 2
    if ntopic == 4: filas = 2;col = 2
    if ntopic == 5: filas = 3;col = 2
    if ntopic == 6: filas = 3;col = 2
    if ntopic == 7: filas = 4;col = 2
    if ntopic == 8: filas = 4;col = 2
    if ntopic == 9: filas = 5;col = 2
    if ntopic == 10: filas = 5;col = 2
    fig, axes = plt.subplots(filas, col, figsize=(10,10), sharey=True, dpi=240)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.5, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, 0.2); ax.set_ylim(0, long_f)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=10)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')
        if i == (ntopic-1):
            break

    fig.tight_layout(w_pad=2)    
    fig.suptitle('Frecuencia de palabras clave por tópico', fontsize=12, y=1.05)    
    fig.savefig("/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/wfreq_temas.jpg")
    


# In[24]:


def nube_palabras(lda_modelo,ntopic):

    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

    cloud = WordCloud(background_color='white',
                      width=2500,
                      height=1800,
                      max_words=10,
                      colormap='tab10',
                      color_func=lambda *args, **kwargs: cols[i],
                      prefer_horizontal=1.0)

    #Está tomando el número de tópicos en lugar del indice de la tabla de lista de modelos
    topics = lda_model.show_topics(formatted=False)
    ntopic = ntopic + 2 
    if ntopic == 1: filas = 1;col = 1
    if ntopic == 2: filas = 1;col = 2
    if ntopic == 3: filas = 2;col = 2
    if ntopic == 4: filas = 2;col = 2
    if ntopic == 5: filas = 3;col = 2
    if ntopic == 6: filas = 3;col = 2
    if ntopic == 7: filas = 4;col = 2
    if ntopic == 8: filas = 4;col = 2
    if ntopic == 9: filas = 5;col = 2
    if ntopic == 10: filas = 5;col = 2
        
    fig, axes = plt.subplots(filas, col, figsize=(10,8), sharex=True, sharey=True)

    for i, ax in enumerate(axes.flatten()):
        fig.add_subplot(ax)
        topic_words = dict(topics[i][1])
        cloud.generate_from_frequencies(topic_words, max_font_size=300)
        plt.gca().imshow(cloud)
        plt.gca().set_title('Topic ' + str(i), fontdict=dict(size=16))
        plt.gca().axis('off')
        if i == (ntopic-1):
            break
    plt.subplots_adjust(wspace=1, hspace=1)
    plt.axis('off')
    plt.margins(x=0, y=0)
    plt.tight_layout()
    #plt.show()
    fig.savefig("/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/nube.jpg")


# In[25]:


def tsne(lda_model,corpus):

    topic_weights = []

    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in row_list])
    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    arr = arr[np.amax(arr, axis=1) > 0.35]

    topic_num = np.argmax(arr, axis=1)

    tsne_model = TSNE(n_components=2, verbose=1, random_state=0, angle=.99, init='pca')
    tsne_lda = tsne_model.fit_transform(arr)

    output_notebook()
    n_topics = 4
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    plot = figure(title="Agrupamiento t-SNE del modelo de temas LDA".format(n_topics), 
                  plot_width=900, plot_height=700)
    plot.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])

    export_png(plot, filename="/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/t-SNE.png")


# In[26]:


#Representar la importancia o aporte de cada clúster al modelo, muestra el conteo de párrafos que están clasificados en cada clúster

def frecuencia_topicos(corpus_filtrado_x):
    fig = plt.figure(figsize=(5,3))
    ax = corpus_filtrado_x['topico'].value_counts().plot(kind='bar')
    plt.show()
    fig.savefig("/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/freq_topic.jpg")


# In[27]:


#Generar el reporte de salida en PDF con el resultado del procesamiento

def reportePDF(contexto,nomarch,palabras_unicasT,palabras_unicasF,relacion_w,cParrafoT,cParrafoF,relacion_p,temas,corpusR,tiempo,print_topic):
    contex = ",".join(contexto['llave'])
    ruta =  "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/reporte1.pdf"
    freq_CsinFiltro = "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/freq_CsinFiltro.jpg"
    freq_CconFiltro = "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/freq_CconFiltro.jpg"
    nube =  "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/nube.jpg"
    wfreq_temas = "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/wfreq_temas.jpg"
    t_SNE="/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/t-SNE.png"
    freq_topic = "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/freq_topic.jpg"
    repxls = "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/corpues.xlsx"

    esp = 12

    doc = SimpleDocTemplate(ruta,pagesize=letter,
                            rightMargin=40,leftMargin=40,
                            topMargin=30,bottomMargin=18)
   
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    styles.add(ParagraphStyle(name='Titulo', alignment=TA_JUSTIFY))

    Texto = [] 
    ptext = '<font size="14">RESULTADOS DEL PROCESAMIENTO AUTOMÁTICO DE TEXTO DEL PLAN DE DESARROLLO TERRITORIAL</font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp*4))
    
    ptext = '<font size="12" color=green>Parametros de entrada</font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp*2))

    ptext = '<font size="10" color=blue>Archivo analizado: </font><font size="10" color=grey>%s</font>' % nomarch
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Palabras clave de búsqueda: </font><font size="10" color=grey>%s</font>' % cadena_interes
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Palabras de contexto asociadas a la búsqueda: </font><font size="10" color=grey>%s</font>' % contex
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp*4))

    ptext = '<font size="12" color=green>Estadísticas generales</font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, 24))

    ptext =  '<font size="10" color=blue>Número de palabras únicas identificadas en el PDT: </font>%s' % len(palabras_unicasT)
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Número de palabras únicas identificadas en el PDT según el contexto de interés: </font>%s' % len(palabras_unicasF)
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Presencia de las palabras del contexto de interés: </font>%s' % str(relacion_w)
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Número de párrafos identificadas en el PDT: </font>%s' % str(cParrafoT)
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Número de párrafos identificadas en el PDT según el contexto de interés: </font>%s' % str(cParrafoF)
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Presencia de párrafos del contexto de interés: </font>%s' % str(relacion_p)
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext = '<font size="10" color=blue>Tiempo de procesamiento del reporte: </font>%s' % str(tiempo)
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp*2))
    
    Texto.append(PageBreak())
    
    ptext =  '<font size="10" color=blue>Top 50 de las palabras más comunes en el PDT:</font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))

    imagen = Image(freq_CsinFiltro, width=500, height=250, hAlign='CENTER')
    Texto.append(imagen)
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Top 50 de las palabras más comunes en el PDT, según el contexto de interés:</font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))

    imagen = Image(freq_CconFiltro, width=500, height=250, hAlign='CENTER')
    Texto.append(imagen)
    Texto.append(Spacer(1, esp))

    Texto.append(PageBreak())

    ptext =  '<font size="12" color=green>Clasificación de tópicos relacionados con el tema de interés: </font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Tópicos identificados en el contexto de interés: </font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10">%s</font>' % temas
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))
    
    imagen = Image(freq_topic, width=500, height=250, hAlign='CENTER')
    Texto.append(imagen)
    Texto.append(Spacer(1, esp))
    
    ptext =  '<font size="10" color=blue>% de presencia de las palabras del contexto según tópico: </font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))
    
    ptext =  '<font size="10">%s</font>' % print_topic
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))
    
    Texto.append(PageBreak())

    ptext =  '<font size="10" color=blue>Nube de palabras de los temas identificados en el contexto de interés: </font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    imagen = Image(nube, hAlign='CENTER')
    Texto.append(imagen)
    Texto.append(Spacer(1, esp))

    Texto.append(PageBreak())

    ptext =  '<font size="10" color=blue>Frecuencia de palabras clave por tema: </font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    imagen = Image(wfreq_temas,width=500, height=350, hAlign='CENTER')
    Texto.append(imagen)
    Texto.append(Spacer(1, esp))

    Texto.append(PageBreak())

    ptext =  '<font size="10" color=blue>Nodos de proximidad t_SNE: </font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))

    imagen = Image(t_SNE, width=500, height=350, hAlign='CENTER')
    Texto.append(imagen)
    Texto.append(Spacer(1, esp))

    ptext =  '<font size="10" color=blue>Visualización de clústeres y frecuencias: </font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))
    
    Texto.append(Paragraph("<font size='8' color=red><link href='http://0.0.0.0:8000/lda_model.html'>Click aquí ver reporte dinámico de los tópicos</link></font>",styles["Normal"]))
    Texto.append(Spacer(1, esp))
    
    ptext =  '<font size="10" color=blue>Consulta de los primeros 20 párrafos relacionados con el contexto de interés:</font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))
    
    corpusRx = corpusR.head(20)
    for index, row in corpusR.iterrows(): 
        ptext =  '<font size="10" color=grey>%s</font>' % row['texto_depurado']
        Texto.append(Paragraph(ptext, styles["Normal"]))
        Texto.append(Spacer(1, esp)) 
 
    corpusHtml = pd.DataFrame(columns=('Topico', 'Texto'))
    corpusHtml['Topico'] = corpusR['topico']
    corpusHtml['Texto'] = corpusR['texto_depurado']
    arch_html = corpusHtml.to_html()
    
    fileHtml = open("/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/corpus_html.html", "w")
    fileHtml.write(arch_html)
    fileHtml.close()

    ptext =  '<font size="10" color=blue>Consultar el contenido relacionado con el tema de interés según tópico:</font>'
    Texto.append(Paragraph(ptext, styles["Normal"]))
    Texto.append(Spacer(1, esp))
  
    Texto.append(Paragraph("<font size='8' color=red><link href='http://0.0.0.0:8000/corpus_html.html'>Click aquí para ver el texto</link></font>",styles["Normal"]))
    Texto.append(Spacer(1, esp))
        
    doc.build(Texto)


# In[28]:


start_time = time.time()
nomarch = "5001_ANT.txt"
cadena_interes = "niñez,adolescencia,juventud,niño,niña,adolescente"
palabrasno_deseadas = "medellin,plan,desarrollar,tu,contar"

#Proceso para convertir los archivos PDF de los planes de desarrollo territoriales a TXT
#Montar acá le proceso en un ciclo para hacer la limpieza por archivo desde que se lee en PDF hasta que se depura
ruta1 = "/Volumes/GoogleDrive/My Drive/TFM/txt/"
ruta2 = "/Volumes/GoogleDrive/My Drive/TFM/txt_utf8/"
ruta3 = "/Volumes/GoogleDrive/My Drive/TFM/corpus/"
ruta4 = "/Volumes/GoogleDrive/My Drive/TFM/rep_pdf/"
#arch = os.listdir(ruta1)

lematizar_pclave()
sword_otrasf()
if len(palabrasno_deseadas):stop_word_extend(palabrasno_deseadas)
tokens = preprocesamiento(nomarch)
##Falta mejorar que reconozca palabras sin acento
## Identificar palabras de contexto relacionadas con el tema de interés
contexto = palabras_clave(cadena_interes)
#print(contexto)
corpus_filtrado = extraccion(tokens,contexto)
#corpus_filtrado.head(25)
palabras_unicasT = freq_corpus(tokens['lemma'],'freq_CsinFiltro','50 palabras más frecuentes en el PDT')
palabras_unicasF = freq_corpus(corpus_filtrado['lemma'],'freq_CconFiltro','50 palabras más frecuentes en el PDT, según el contexto buscado')
relacion_w = round((len(palabras_unicasF)/len(palabras_unicasT))*100,2)
relacion_p = round((len(corpus_filtrado['lemma'])/len(tokens['lemma']))*100,2)
cParrafoT = len(tokens['lemma'])
cParrafoF = len(corpus_filtrado['lemma'])
id2word = corpora.Dictionary(corpus_filtrado['lemma'])
id2word.filter_extremes(no_below=2, no_above=0.97, keep_n=None)
corpus = [id2word.doc2bow(text) for text in corpus_filtrado['lemma']]
#revisar parámetros para replicar las salidas
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=corpus_filtrado['lemma'],limit=10, start=2, step=1)
start=2
limit=10
step=1
x = range(start, limit, step)
#for m, cv in zip(x, coherence_values):
#    print("Num Topics =", m, " has Coherence Value of", round(cv, 2))
cvd = pd.DataFrame({"topic": x, "cv" : coherence_values}) 
cvd["cv"] = round(cvd["cv"],2)
cvf = cvd[cvd['cv'] == cvd["cv"].max()]
ntopic = int(cvf["topic"].head(1)) - 2
lda_model = model_list[ntopic]
pprint(lda_model.print_topics()) 
print_topic = str(lda_model.print_topics()).replace('(','<br />\n<br />\n(')
temas = display_topics(lda_model)
corpus_filtrado_x = mapeo_topico(lda_model,corpus_filtrado)
frecuencia_topicos(corpus_filtrado_x)
davis(lda_model, corpus, id2word)
long_f = 20
weight = 0.2
frecuencias_palabras(lda_model,ntopic,long_f,weight)
nube_palabras(lda_model,ntopic)
tsne(lda_model,corpus)
end_time = time.time()
tiempo = round((end_time-start_time)/60,2)
reportePDF(contexto,nomarch,palabras_unicasT,palabras_unicasF,relacion_w,cParrafoT,cParrafoF,relacion_p,temas,corpus_filtrado_x,tiempo,print_topic)  
reportePDF(contexto,nomarch,palabras_unicasT,palabras_unicasF,relacion_w,cParrafoT,cParrafoF,relacion_p,temas,corpus_filtrado_x,tiempo,print_topic)  

