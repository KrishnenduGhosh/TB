import os
import word2vec
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import nltk
from lib.textsplit.tools import get_penalty, get_segments
from lib.textsplit.algorithm import split_optimal, split_greedy, get_total
import re

def section(subject,segment_len,sentence_analyzer,wrdvecs):
	dir_name='2_Text/'+subject
	dir_out_path='3_Section/'+subject
	if not os.path.exists(dir_out_path):
		os.makedirs(dir_out_path)
	for book_name in os.listdir(dir_name):
		book_path=dir_name+'/'+book_name
		out_path = dir_out_path+'/'+book_name+"/"
		if not os.path.exists(out_path):
			os.makedirs(out_path)
		print("book_path: " + book_path)
		with open(book_path, 'rt') as f:
			text = " ".join(f.readlines()).replace('', '')
			text = text.replace('[\W_]+', '').replace('  ', ' ')
			text = text.strip().replace('()', '').replace('\n',' ').replace("‘", "'").replace("’", "'")
		sentenced_text = sentence_analyzer.tokenize(text)
		vecr = CountVectorizer(vocabulary=wrdvecs.index)
		sentence_vectors = vecr.transform(sentenced_text).dot(wrdvecs)
		if(len(sentence_vectors) > 40):
			penalty = get_penalty([sentence_vectors], segment_len)
			optimal_segmentation = split_optimal(sentence_vectors, penalty, seg_limit=250)
			segmented_text = get_segments(sentenced_text, optimal_segmentation)
			for i, segment_sentences in enumerate(segmented_text):
				segment_str = ''.join(segment_sentences).replace(".", ". ").replace("?", "? ").replace("\\s+", "\\s")
				with open(out_path+str(i)+'.txt', "w") as text_file:
					text_file.write("%s" % segment_str.strip())
			greedy_segmentation = split_greedy(sentence_vectors, max_splits=len(optimal_segmentation.splits))
			greedy_segmented_text = get_segments(sentenced_text, greedy_segmentation)
			lengths_optimal = [len(segment) for segment in segmented_text for sentence in segment]
			lengths_greedy = [len(segment) for segment in greedy_segmented_text for sentence in segment]
			df = pd.DataFrame({'greedy':lengths_greedy, 'optimal': lengths_optimal})
			totals = [get_total(sentence_vectors, seg.splits, penalty) for seg in [optimal_segmentation, greedy_segmentation]]

if __name__=="__main__":
	corpus_path = 'lib/text8'  # be sure your corpus is cleaned from punctuation and lowercased
	if not os.path.exists(corpus_path):
		get_ipython().system(u'wget http://mattmahoney.net/dc/text8.zip')
		get_ipython().system(u'unzip {corpus_path}')
	wrdvec_path = 'lib/wrdvecs-text8.bin'
	if not os.path.exists(wrdvec_path):
		get_ipython().magic(u"time word2vec.word2vec(corpus_path, wrdvec_path, cbow=1, iter_=5, hs=1, threads=4, sample='1e-5', window=15, size=200, binary=1)")
	model = word2vec.load(wrdvec_path)
	wrdvecs = pd.DataFrame(model.vectors, index=model.vocab)
	filename = "lib/finalized_model.sav"
	pickle.dump(model, open(filename, 'wb'))
	del model
	print(wrdvecs.shape)
	nltk.download('punkt')
	sentence_analyzer = nltk.data.load('tokenizers/punkt/english.pickle')

	segment_len = 20  # segment target length in sentences

	subjects = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	#subjects = ['test']
	for subject in subjects:
		print(subject)
		section(subject,segment_len,sentence_analyzer,wrdvecs)
	print("TXTs are segmented into Sections ...........")
