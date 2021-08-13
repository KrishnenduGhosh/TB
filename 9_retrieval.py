import warnings
warnings.filterwarnings("ignore")
import os
import sys
import time
import subprocess
import xml.dom.minidom
import re
import nltk
from nltk import word_tokenize
from nltk.util import ngrams
from nltk.translate import IBMModel1, AlignedSent, Alignment
from rank_bm25 import BM25Okapi
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import scoring
from sklearn.feature_extraction.text import CountVectorizer
from math import sqrt
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import json
import word2vec
import pandas as pd
import pickle
import csv
import spacy
import math
import tagme
from random import seed
from random import randrange
from random import random
from csv import reader
from math import exp
from sklearn import metrics

def load_csv(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			rows = list()
			for i in range(len(row)):
				if i < len(row)-1:
					rows.append(float(row[i]))
				else:
					rows.append(int(float(row[i])))
			dataset.append(rows)
	return dataset

def load_csv_test(filename):
	dataset = list()
	with open(filename, 'r') as file:
		csv_reader = csv.reader(file)
		for row in csv_reader:
			rows = list()
			for i in range(len(row)):
				rows.append(float(row[i]))
			dataset.append(rows)
	return dataset

def str_column_to_float(dataset, column):
	fdata = []
	for row in dataset:
		fdata[column] = float(row[column].strip())
		return fdata

def str_column_to_int(dataset, column):
	class_values = [row[column] for row in dataset]
	unique = set(class_values)
	lookup = dict()
	for i, value in enumerate(unique):
		lookup[value] = i
	for row in dataset:
		row[column] = lookup[row[column]]
	return lookup

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			if minmax[i][1] != minmax[i][0]:
				row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])
			else:
				row[i] = 0.0

def cross_validation_split(dataset, n_folds):
	dataset_split = list()
	dataset_copy = list(dataset)
	fold_size = int(len(dataset) / n_folds)
	for i in range(n_folds):
		fold = list()
		while len(fold) < fold_size:
			index = randrange(len(dataset_copy))
			fold.append(dataset_copy.pop(index))
		dataset_split.append(fold)
	return dataset_split

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
	folds = cross_validation_split(dataset, n_folds)
	scores = list()
	for fold in folds:
		train_set = list(folds)
		train_set.remove(fold)
		train_set = sum(train_set, [])
		test_set = list()
		for row in fold:
			row_copy = list(row)
			test_set.append(row_copy)
			row_copy[-1] = None
		predicted = algorithm(train_set, test_set, *args)
		actual = [row[-1] for row in fold]
		accuracy = metrics.accuracy_score(actual, predicted)
		scores.append(accuracy)
	return scores

def activate(weights, inputs):
	activation = weights[-1]
	for i in range(len(weights)-1):
		activation += weights[i] * inputs[i]
	return activation

def transfer(activation):
	return 1.0 / (1.0 + math.exp(-activation))

def forward_propagate(network, row):
	inputs = row
	for layer in network:
		new_inputs = []
		for neuron in layer:
			activation = activate(neuron['weights'], inputs)
			neuron['output'] = transfer(activation)
			new_inputs.append(neuron['output'])
		inputs = new_inputs
	return inputs

def transfer_derivative(output):
	return output * (1.0 - output)

def backward_propagate_error(network, expected):
	for i in reversed(range(len(network))):
		layer = network[i]
		errors = list()
		if i != len(network)-1:
			for j in range(len(layer)):
				error = 0.0
				for neuron in network[i + 1]:
					error += (neuron['weights'][j] * neuron['delta'])
				errors.append(error)
		else:
			for j in range(len(layer)):
				neuron = layer[j]
				errors.append(expected[j] - neuron['output'])
		for j in range(len(layer)):
			neuron = layer[j]
			neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])

def update_weights(network, row, l_rate):
	for i in range(len(network)):
		inputs = row[:-1]
		if i != 0:
			inputs = [neuron['output'] for neuron in network[i - 1]]
		for neuron in network[i]:
			for j in range(len(inputs)):
				neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
			neuron['weights'][-1] += l_rate * neuron['delta']

def train_network(network, train, l_rate, n_epoch, n_outputs):
	for epoch in range(n_epoch):
		for row in train:
			outputs = forward_propagate(network, row)
			expected = [0 for i in range(n_outputs)]
			if row[-1] < len(expected):
				expected[row[-1]] = 1
			backward_propagate_error(network, expected)
			update_weights(network, row, l_rate)

def initialize_network(n_inputs, n_hidden, n_outputs):
	network = list()
	hidden_layer = [{'weights':[random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
	network.append(hidden_layer)
	output_layer = [{'weights':[random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
	network.append(output_layer)
	return network

def predict(network, row):
	outputs = forward_propagate(network, row)
	return outputs.index(max(outputs))

def back_propagation(train, test, l_rate, n_epoch, n_hidden):
	n_inputs = len(train[0]) - 1
	n_outputs = len(set([row[-1] for row in train]))
	network = initialize_network(n_inputs, n_hidden, n_outputs)
	train_network(network, train, l_rate, n_epoch, n_outputs)
	predictions = list()
	for row in test:
		prediction = predict(network, row)
		predictions.append(prediction)
	return(predictions)

def cleanhtml(raw_html):
	cleanr = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
	cleantext = re.sub(cleanr, '', raw_html)
	return cleantext

def selectQA(t): # Preprocesses <QA>.xml to <QA>.txt in form: (Qid,Qtitle,Qbody,Qanswer,Qtags)
	if not os.path.isfile('QA/'+t+'_final.xml'):
		Q={}
		A={}
		rt = t
		if t == 'Science':
			rt = 'Physics'
		doc = xml.dom.minidom.parse('QA/'+rt+'.xml')
		rows = doc.getElementsByTagName('row')
		for row in rows:
			if row.attributes['PostTypeId'].value == '1':
				key=row.attributes['Id'].value
				if not row.getElementsByTagName('AcceptedAnswerId'):
					val=process1(row.attributes['Title'].value)+"#"+process1(row.attributes['Body'].value)+"#"+process2(row.attributes['Tags'].value)
					Q[key]=val.replace("\n","").replace("\n","")
				else:
					val=process1(row.attributes['Title'].value)+"#"+process1(row.attributes['Body'].value)+"#"+process2(row.attributes['Tags'].value)+"#"+row.attributes['AcceptedAnswerId'].value
					Q[key]=val.replace("\n","").replace("\n","")
			elif row.attributes['PostTypeId'].value == '2':
				key=row.attributes['Id'].value
				val=process1(row.attributes['Body'].value)+"#"+row.attributes['Score'].value+"#"+row.attributes['ParentId'].value
				A[key]=val.replace("\n","").replace("\n","")
			else:
				pass
		for k in Q.keys():
			vpart = Q[k].split("#")  # vpart[3] > AcceptedAnswerId
			if len(vpart) == 3: # Question has no AcceptedAnswerId
				pass
			else: # Question has AcceptedAnswerId
				Q[k]=vpart[0]+"#"+vpart[1]+"#"+vpart[2]+"#"+A[vpart[3]].split("#")[0]+"#"+A[vpart[3]].split("#")[1]
		for k in A.keys():
			vpart = A[k].split("#") # vpart[2] > ParentId
			Qpart = Q[vpart[2]].split("#")
			if len(Qpart) == 3: # Question has no answer
				Q[vpart[2]]=Q[vpart[2]]+"#"+vpart[0]+"#"+vpart[1]
			elif len(Qpart) == 4: # Question has answer with no score
				eans=Qpart[3]
				eans_score=0
				ans=vpart[0]
				ans_score=vpart[1]
				if int(eans_score) < int(ans_score):
					#val=Qpart[0]+"#"+Qpart[1]+"#"+Qpart[2]+"#"+ans+"#"+ans_score
					val=Qpart[0]+"#"+Qpart[1]+"#"+Qpart[2]+"#"+ans
					Q[vpart[2]]=val
			else: # Question has answer with score
				eans=Qpart[3]
				eans_score=Qpart[4]
				ans=vpart[0]
				ans_score=vpart[1]
				if int(eans_score) < int(ans_score):
					#val=Qpart[0]+"#"+Qpart[1]+"#"+Qpart[2]+"#"+ans+"#"+ans_score
					val=Qpart[0]+"#"+Qpart[1]+"#"+Qpart[2]+"#"+ans
					Q[vpart[2]]=val
		f=open('QA/QA.txt','a')
		for k in Q.keys():
			f.write(k+'#'+Q[k]+'\n')
		f.close()

def listToString(s):
	str1 = ""
	for ele in s:
		str1 += ele
	return str1

def load_wv():
	wrdvec_path = './lib/wrdvecs-text8.bin'
	model = word2vec.load(wrdvec_path)
	wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
	filename = "./lib/finalized_model.sav"
	pickle.dump(model, open(filename, 'wb'))
	del model
	sentence_analyzer = nltk.data.load('tokenizers/punkt/english.pickle')
	return wrdvecs,sentence_analyzer

def get_wv(line,wrdvecs,sentence_analyzer): # 
	sentenced_text = sentence_analyzer.tokenize(line)
	vecr = CountVectorizer(vocabulary=wrdvecs.index)
	sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
	wv = [0] * 200
	for sv in sentence_vectors:
		for i in range(len(sv)):
			wv[i] += sv[i]
	for i in range(len(wv)):
		if len(sentence_vectors) > 0:
			wv[i] = round(float(wv[i]/len(sentence_vectors)),2)
		else:
			wv[i] = 0.0
	return wv

def get_f1(line1,line2): # Feature 1,2,3: Word n-gram (n = 1,2,3) overlap
	token1 = word_tokenize(line1)
	token2 = word_tokenize(line2)
	qgram1 = len(list(ngrams(token1,1)))
	qgram2 = len(list(ngrams(token1,2)))
	qgram3 = len(list(ngrams(token1,3)))
	onegram = len([value for value in list(ngrams(token1,1)) if value in list(ngrams(token2,1))])
	twogram = len([value for value in list(ngrams(token1,2)) if value in list(ngrams(token2,2))])
	threegram = len([value for value in list(ngrams(token1,3)) if value in list(ngrams(token2,3))])
	gram1 = 0.0
	gram2 = 0.0
	gram3 = 0.0
	if qgram1 != 0.0:
		gram1 = round(float(onegram/qgram1),4)
	if qgram2 != 0.0:
		gram2 = round(float(twogram/qgram2),4)
	if qgram3 != 0.0:
		gram3 = round(float(threegram/qgram3),4)
	f = str(gram1)+","+str(gram2)+","+str(gram3)
	return f

def get_f2(line1,line2,corpus): # Feature 4: BM25 score
	bm25_score = 0.0
	tokenized_corpus = [doc.split(" ") for doc in corpus]
	bm25 = BM25Okapi(tokenized_corpus)
	tokenized_query = line1.split(" ")
	doc_scores = bm25.get_scores(tokenized_query)
	for i in range(len(corpus)):
		if line2 == corpus[i]:
			bm25_score=doc_scores[i]
	f = str(round(bm25_score,4))
	return f

def get_f4(line1,line2): # Feature 8: Noun overlap
	nouns1 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line1))):
		if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
			nouns1.append(word)
	nouns2 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line2))):
		if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
			nouns2.append(word)
	no = [value for value in nouns1 if value in nouns2]
	f = "0.0"
	if len(nouns1) != 0:
		f = ""+str(round(float(len(no)/len(nouns1)),4))
	return f

def get_f5(line1,line2): # Feature 9: Verb overlap
	verbs1 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line1))):
		if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
			verbs1.append(word)
	verbs2 = []
	for word,pos in nltk.pos_tag(nltk.word_tokenize(str(line2))):
		if (pos == 'VB' or pos == 'VBD' or pos == 'VBG' or pos == 'VBN' or pos == 'VBP' or pos == 'VBZ'):
			verbs2.append(word)
	vo = [value for value in verbs1 if value in verbs2]
	f = "0.0"
	if len(verbs1) != 0:
		f = ""+str(round(float(len(vo)/len(verbs1)),4))
	return f

def get_f6(line1,line2): # Feature 10: Dependency pair overlap
	nlp = spacy.load("en_core_web_sm")
	dep1 = []
	for token in nlp(line1):
		dep1.append(token.text+"<"+token.dep_+"<"+token.head.text)
	dep2 = []
	for token in nlp(line2):
		dep2.append(token.text+"<"+token.dep_+"<"+token.head.text)
	dpo = [value for value in dep1 if value in dep2]
	f = "0.0"
	if len(dep1) != 0:
		f = ""+str(round(float(len(dpo)/len(dep1)),4))
	return f

def get_f7(line1,line2): # Feature 11: Named entity overlap
	ne1=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(str(line1))))
	ne2=nltk.ne_chunk(nltk.pos_tag(nltk.word_tokenize(str(line2))))
	neo = [value for value in ne1 if value in ne2]
	f = "0.0"
	if len(ne1) != 0:
		f = ""+str(round(float(len(neo)/len(ne1)),4))
	return f

def get_f31(line1,line2,wrdvecs,sentence_analyzer): # Feature 10
	wv1 = get_wv(line1,wrdvecs,sentence_analyzer)
	wv2 = get_wv(line2,wrdvecs,sentence_analyzer)
	top = 0.0
	for i in range(len(wv1)):
		top = top + (round(wv1[i],2) * round(wv2[i],2))
	b1 = 0.0
	for i in range(len(wv1)):
		#print(wv1[i])
		b1 = b1 + (round(wv1[i],2) * round(wv1[i],2))
	b2 = 0.0
	for i in range(len(wv1)):
		#print(wv2[i])
		b2 = b2 + (round(wv2[i],2) * round(wv2[i],2))
	bottom = sqrt(b1) * sqrt(b2)
	if bottom == 0.0:
		cosim = 0.0
	else:
		cosim = float(top/bottom)
	#print(cosim)
	f = ""+str(round(cosim,4))
	return f

def get_f32(line1,line2,wrdvecs,sentence_analyzer): # Feature 11
	st = PorterStemmer()
	sline1 = ''
	sline2 = ''
	for w in line1.split(" "):
		sline1 = sline1 + ' ' + st.stem(w)
	for w in line2.split(" "):
		sline2 = sline2 + ' ' + st.stem(w)
	sline1 = sline1.replace('  ',' ')
	sline2 = sline2.replace('  ',' ')
	wv1 = get_wv(sline1,wrdvecs,sentence_analyzer)
	wv2 = get_wv(sline2,wrdvecs,sentence_analyzer)
	top = 0.0
	for i in range(len(wv1)):
		top = top + (round(wv1[i],2) * round(wv2[i],2))
	b1 = 0.0
	for i in range(len(wv1)):
		#print(wv1[i])
		b1 = b1 + (round(wv1[i],2) * round(wv1[i],2))
	b2 = 0.0
	for i in range(len(wv1)):
		#print(wv2[i])
		b2 = b2 + (round(wv2[i],2) * round(wv2[i],2))
	bottom = sqrt(b1) * sqrt(b2)
	if bottom == 0.0:
		cosim = 0.0
	else:
		cosim = float(top/bottom)
	#print(cosim)
	f = ""+str(round(cosim,4))
	return f

def get_f33(line1,line2,wrdvecs,sentence_analyzer): # Feature 12
	sline1 = ''
	sline2 = ''
	for w in line1.split(" "):
		if w not in stopwords.words('english'):
			sline1 = sline1 + ' ' + w
	for w in line2.split(" "):
		if w not in stopwords.words('english'):
			sline2 = sline2 + ' ' + w
	sline1 = sline1.replace('  ',' ')
	sline2 = sline2.replace('  ',' ')
	wv1 = get_wv(sline1,wrdvecs,sentence_analyzer)
	wv2 = get_wv(sline2,wrdvecs,sentence_analyzer)
	top = 0.0
	for i in range(len(wv1)):
		top = top + (round(wv1[i],2) * round(wv2[i],2))
	b1 = 0.0
	for i in range(len(wv1)):
		#print(wv1[i])
		b1 = b1 + (round(wv1[i],2) * round(wv1[i],2))
	b2 = 0.0
	for i in range(len(wv1)):
		#print(wv2[i])
		b2 = b2 + (round(wv2[i],2) * round(wv2[i],2))
	bottom = sqrt(b1) * sqrt(b2)
	if bottom == 0.0:
		cosim = 0.0
	else:
		cosim = float(top/bottom)
	#print(cosim)
	f = ""+str(round(cosim,4))
	return f

def get_feature(line1,line2,corpus,vecr,sentence_analyzer): # generate features for <query,QA pair>
	f1 = get_f1(line1,line2)
	f2 = get_f2(line1,line2,corpus)
	f31 = get_f31(line1,line2,vecr,sentence_analyzer)
	f32 = get_f32(line1,line2,vecr,sentence_analyzer)
	f33 = get_f33(line1,line2,vecr,sentence_analyzer)
	f4 = get_f4(line1,line2)
	f5 = get_f5(line1,line2)
	f6 = get_f6(line1,line2)
	f7 = get_f7(line1,line2)
	f8 = get_f8(line1,line2)
	f9 = get_f9(line1,line2)
	feature = str(f1)+","+str(f2)+","+str(f31)+","+str(f32)+","+str(f33)+","+str(f4)+","+str(f5)+","+str(f6)+","+str(f7)+","+str(f8)+","+str(f9)
	return feature

def build_index(t): # create corpus for BM25
	corpus = []
	with open('7_cFeature/'+t+'.json') as json_file:
		data = json.load(json_file)
		for d in data:
			corpus.append(d['text'])
	return corpus

def get_f8(line1,line2): # Feature 12: Word alignment
	prob = {}
	f8=open('10_Retrieval/Parallel_Corpora/Translation.txt','r')
	for l in f8:
		lpart = l.strip().split("\t")
		print(l.strip())
		print(len(lpart))
		prob[lpart[0]+'\t'+lpart[1]]=lpart[2]
	f8.close()
	token1 = word_tokenize(line1)
	token2 = word_tokenize(line2)
	ctr = 0
	total = 0.0
	for t1 in token1:
		for t2 in token2:
			if t1+'\t'+t2 in prob.keys():
				total+= prob[t1+'\t'+t2]
			else:
				total+= 0.0
			ctr+=1
	if ctr > 1:
		return "12:"+str(round((total/ctr),2))
	else:
		return "12:0.0"

def get_f9(line1,line2): # Feature 9: Frame overlap
	parser = sling.Parser("lib/caspar.flow")
	doc1 = parser.parse(line1)
	doc2 = parser.parse(line2)
	#print(doc1.frame.data(pretty=True))
	#print(doc2.frame.data(pretty=True))
	#for m1 in doc1.mentions:
	#	print("mention", doc1.phrase(m1.begin, m1.end))
	frame1 = frame2 = []
	for m1 in doc1.mentions:
		frame1.append(doc1.phrase(m1.begin, m1.end))
	for m2 in doc2.mentions:
		frame2.append(doc2.phrase(m2.begin, m2.end))
	fo = [value for value in ne1 if value in ne2]
	f = "13:"+str(len(fo))
	return f

def build_pc(): # Create two parallel corpora from AskUbuntu & SemEal
	files=['AskUbuntu','SemEval']
	for f in files:
		print("Processing corpus: ",f)
		GS = {}
		q = {}
		Q = {}
		f1=open('10_Retrieval/Parallel_Corpora/'+f+'/GS.txt','r')
		for line1 in f1:
			if line1.split(" ")[0] in GS.keys():
				val = GS[line1.split(" ")[0]]
				val.append(line1.split(" ")[2])
				GS[line1.split(" ")[0]]=val
			else:
				val = []
				val.append(line1.split(" ")[2])
				GS[line1.split(" ")[0]]=val
		f1.close()
		f2=open('10_Retrieval/Parallel_Corpora/'+f+'/Query.txt','r')
		for line2 in f2:
			q[line2.split("\t")[0]]=line2.split("\t")[1]+"\t"+line2.split("\t")[2]
		f2.close()
		f3=open('10_Retrieval/Parallel_Corpora/'+f+'/Question.txt','r')
		for line3 in f3:
			Q[line3.split("\t")[0]]=line3.split("\t")[1]+"\t"+line3.split("\t")[2]
		f3.close()
		for k in GS.keys():
			for e in GS[k]:
				one=q[k].split("\t")[0]+"\t"+Q[e].split("\t")[0]
				two=q[k].split("\t")[1]+"\t"+Q[e].split("\t")[1]
				fw=open('10_Retrieval/Parallel_Corpora/Corpus.txt','a')
				fw.write(one+"\n")
				fw.write(two+"\n")
				fw.close()
	print("Created parallel corpora from AskUbuntu & SemEal as Corpus.txt")

def build_IBM1(): # Train a translation model based on Corpus.txt
	bitext = []
	words = []
	f=open('10_Retrieval/Parallel_Corpora/Corpus.txt','r')
	for line in f:
		lpart=line.split("\t")
		token1 = word_tokenize(lpart[0])
		token2 = word_tokenize(lpart[1])
		for t1 in token1:
			if t1 not in words:
				words.append(t1)
		for t2 in token2:
			if t2 not in words:
				words.append(t2)
		bitext.append(AlignedSent(token1, token2))
	f.close()
	print("Parallel corpora loaded ....")
	ibm1 = IBMModel1(bitext, 5)
	print("Translation model built ....")
	f=open('10_Retrieval/Parallel_Corpora/Translation.txt','a')
	for w1 in words:
		for w2 in words:
			f.write(w1+'\t'+w1+'\t'+str(round(ibm1.translation_table[w1][w2], 4))+'\n')
	f.close()
	print("Parallel corpora loaded ....")

def dnn(file): # DNN performance
	dataset = load_csv(file)
	minmax = dataset_minmax(dataset)
	normalize_dataset(dataset, minmax)

	n_folds = 5
	l_rate = 0.3
	n_epoch = 5
	n_hidden = 1
	scores = evaluate_algorithm(dataset, back_propagation, n_folds, l_rate, n_epoch, n_hidden)
	print('len(scores): %s' % len(scores))
	print('Scores: %s' % scores)
	print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))

	l_rate = 0.3
	n_epoch = 1
	n_hidden = 5
	predicted = back_propagation(dataset, dataset, l_rate, n_epoch, n_hidden)
	print('len(predicted): %s' % len(scores))
	return predicted

def create_qQ(t): #
	corpus = build_index(t)
	wrdvecs,sentence_analyzer = load_wv()

	if not os.path.isfile('10_Retrieval/Parallel_Corpora/Corpus.txt'):
		build_pc()
	if not os.path.isfile('10_Retrieval/Parallel_Corpora/Translation.txt'):
		build_IBM1()

	qQlist = []
	f=open('GS/GS_Retrieval.txt','r')
	for l in f:
		lpart = l.strip().split(" ")
		qQlist.append(lpart[2])
	f.close()

	if not os.path.isfile('10_Retrieval/qQ.csv'):
		fq=open('9_Query/'+t+'.txt','r')
		for lq in fq:
			qqpart=lq.strip().split("\t")
			lineq=qqpart[1].replace("_"," ")
			fQ=open('QA/QA.txt','r')
			for lQ in fQ:
				Qpart=lQ.strip().split("\t")
				lineQ=Qpart[1]
				feats=get_feature(lineq,lineQ,corpus,wrdvecs,sentence_analyzer)
				key = qqpart[0]+'_'+qqpart[1]+'_'+Qpart[0]
				if key in qQlist:
					fw=open('10_Retrieval/qQ.csv','a')
					fw.write(feats+',1\n')
					fw.close()
					fw=open('10_Retrieval/qQ.txt','a')
					fw.write(qqpart[0]+'_'+qqpart[1]+' > '+key+' 1\n')
					fw.close()
				else:
					fw=open('10_Retrieval/qQ.csv','a')
					fw.write(feats+',0\n')
					fw.close()
					fw=open('10_Retrieval/qQ.txt','a')
					fw.write(qqpart[0]+'_'+qqpart[1]+' > '+key+' 0\n')
					fw.close()
			fQ.close()
		fq.close()
	print("qQ.txt stored ....")

def retrieve(): # initial retrieval
	topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	for t in topics:
		#print(t)
		create_qQ(t)
	
	predicted = dnn('10_Retrieval/qQ.csv')
	#print(len(predicted))
	# write > write top 10 QA pairs > RT.txt
	rlist = []
	f=open('10_Retrieval/qQ.txt','r')
	i = 0
	for line in f:
		lpart = line.strip().split(" ")
		if lpart[2] not in rlist and str(predicted[i]) == '1':
			fw=open('10_Retrieval/RT1.txt','a')
			fw.write(lpart[0]+' > '+lpart[2]+' '+str(predicted[i])+" 1 RT\n")
			fw.close()
			rlist.append(lpart[2])
		if lpart[2] not in rlist and str(predicted[i]) == '0':
			fw=open('10_Retrieval/RT1.txt','a')
			fw.write(lpart[0]+' > '+lpart[2]+' '+str(predicted[i])+" 0 RT\n")
			fw.close()
			rlist.append(lpart[2])
		i+=1
	print(i)
	f.close()

	line_dict = {}
	label_dict = {}
	fr=open('10_Retrieval/RT1.txt','r')
	for line in fr:
		lpart = line.strip().split(" ")
		temp_line = []
		temp_label = []
		if lpart[0] in line_dict.keys():
			temp_line = line_dict[lpart[0]]
			temp_line.append(line.strip())
			line_dict[lpart[0]] = temp_line
		else:
			temp_line = []
			temp_line.append(line.strip())
			line_dict[lpart[0]] = temp_line
		if lpart[0] in label_dict.keys():
			temp_label = label_dict[lpart[0]]
			temp_label.append(lpart[3])
			label_dict[lpart[0]] = temp_label
		else:
			temp_label = []
			temp_label.append(lpart[0])
			label_dict[lpart[0]] = temp_label
	fr.close()

	for kk in line_dict.keys():
		temp_line = line_dict[kk]
		temp_label = label_dict[kk]
		clist = [temp_line for _,temp_line in sorted(zip(temp_label,temp_line))]
		clist.reverse()
		for cline in clist:
			fw=open('10_Retrieval/RT.txt','a')
			fw.write(cline+"\n")
			fw.close()

def sup_s(Qi,Q_init,Qis): # support scoring for a Q in Q_init
	QQ = []
	QQs = []
	Qfs = float(Qis)
	fr2 = open('QA/BM25_Q.txt','r')
	for line in fr2:
		lpart = line.strip().split("\t")
		if lpart[0] == Qi:
			Q = lpart[1].replace("[","").replace("]","").replace("\'\'","")
			QQ = Q.split(", ")
			Qs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
			QQs = Qs.split(", ")
			break
	fr2.close()

	temp_s = 0.0
	for q in Q_init:
		for i in range(0,len(QQ)):
			if q == QQ[i]:
				temp_s += QQs[i]
	Qfs *= temp_s
	return Qfs

def cum_s(Qi,Q_init,Qis): # support scoring for a Q in Q_init
	QQ = []
	QQs = []
	Qfs = Qis
	fr2 = open('QA/BM25_Q.txt','r')
	for line in fr2:
		lpart = line.strip().split("\t")
		if lpart[0] == Q and Q in Q_init:
			Q = lpart[1].replace("[","").replace("]","").replace("\'\'","")
			QQ = Q.split(", ")
			Qs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
			QQs = Qs.split(", ")
			continue
	fr2.close()

	cQQs = []
	for i in range(0,len(QQ)):
		temps = 0.0
		fr2 = open('QA/BM25_Q.txt','r')
		for line in fr2:
			lpart = line.strip().split("\t")
			if lpart[0] == QQ[i] and QQ[i] in Q_init:
				cQs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
				cum_Qs = cQs.split(", ")
				for s in cum_Qs:
					temps += s
				break
		cQQs[i] = temps
		fr2.close()

	temp_s = 0.0
	for i in range(0,len(QQ)):
		if q == QQ[i]:
			temp_s += cQQs[i]
	Qfs *= temp_s
	return Qfs

def cscore(Q_init,Q_init_s): # cumulative scoring for all Q_init
	Q_final = []
	Q_final_s = []
	for i in range(0,len(Q_init)):
		Qi = Q_init[i]
		Qis = Q_init_s[i]
		Qfs = sup_s(Qi,Q_init,Qis)
		#Qfs = cum_s(Qi,Q_init,Qis)
		Q_final.append(Qi.replace("\'",""))
		Q_final_s.append(Qis.replace("\'",""))
	return  Q_final, Q_final_s

def rerank(): # reranking
	fr1 = open('QA/BM25_qQ.txt','r')
	for line in fr1:
		lpart = line.strip().split("\t")
		q = lpart[1].replace("[","").replace("]","").replace("\'\'","")
		Q_init = q.split(", ")
		qs = lpart[2].replace("[","").replace("]","").replace("\'\'","")
		Q_init_s = qs.split(", ")
		Q_final, Q_final_s = cscore(Q_init,Q_init_s)
		fw=open('QA/RR_q.txt','a')
		fw.write(str(lpart[0]).replace(" ","_")+"\t"+str(Q_final)+"\t"+str(Q_final_s)+"\n")
		fw.close()
		for i in range(0,len(Q_final)):
			fw=open('10_Retrieval/RR.txt','a')
			fw.write(str(lpart[0]).replace(" ","_")+" > "+lpart[0].replace(" ","_")+"_"+str(Q_final[i]).replace("\'","")+" 1 "+str(Q_final_s[i])+" RR\n")
			fw.close()

def gradify(): # filtering QAs to remove QAs if not associated with concerned grade-level

	f2=open('10_Retrieval/RR.txt','r')
	for l2 in f2:
		lpart = l2.strip().split(" ")
		grade = lpart[2].split("_")[1]
		#print(grade)
		Qid = lpart[2].split("_")[-1]
		#print(Qid)

		clist_g = []
		f3=open('4_Concept/'+grade+'.txt','r')
		for l3 in f3:
			clist_g.append(l3.strip())
		f3.close()

		QA = ""
		f4=open('QA/QA.txt','r')
		for l4 in f4:
			lpart = l4.strip().split("\t")
			if lpart[0] == Qid:
				QA = lpart[1]
				break
		f4.close()
		
		topics_list=[]
		f4=open('QA/QA_Concepts.txt','r')
		for l5 in f5:
			lpart = l5.strip().split("\t")
			if lpart[0] == Qid:
				Q = lpart[1].replace("[","").replace("]","").replace("\'\'","")
				topics_list = Q.split(", ")
				break
		f5.close()

		ctr = 0
		for topic in topics_list:
			if topic in clist_g:
				ctr += 1
		fw=open('10_Retrieval/AUG.txt','a')
		fw.write(str(l2.strip())+"\n")
		fw.close()
	f2.close()

def get_QA_concepts():
		QA = ""
		f4=open('QA/QA.txt','r')
		for l4 in f4:
			lpart = l4.strip().split("\t")
			topics_list=[]
			data=lpart[1].encode().decode('utf-8').replace('\n', '').lower()
			resp = tagme.annotate(data)
			if (resp!=None and resp.get_annotations(0.4)!=None):
				for ann in resp.get_annotations(0.4):
					if ann.mention not in mention_list:
						topics_list.append(ann.entity_title.lower())
			
			fw=open('QA/QA_Concepts.txt','a')
			fw.write(str(lpart[0]) +"\t"+ str(topics_list))
			fw.close()
		f4.close()

def evaluate(): # evaluate retrieval, reranking models
	cmd1 = 'trec_eval/trec_eval -M10 GS/GS_Retrieval.txt 10_Retrieval/RT.txt'
	cmd2 = 'trec_eval/trec_eval -M10 GS/GS_Retrieval.txt 10_Retrieval/RR.txt'
	cmd3 = 'trec_eval/trec_eval -M10 GS/GS_Retrieval.txt 10_Retrieval/AUG.txt'
	os.system(cmd1)
	os.system(cmd2)
	os.system(cmd3)

def main(): #main for menu
	start_time = time.time()
	topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	#topics = ['test']
	for t in topics:
		print(t)
		selectQA(t)

	retrieve()
	rerank()
	
	get_QA_concepts()
	gradify()
	gradify()
	evaluate()
	print("Exuction time: ", (time.time() - start_time))

if __name__=="__main__":
	tagme.GCUBE_TOKEN = "0b4eed68-e456-4488-a5a6-7a608ea7e32b-843339462"
	main()
