import warnings
warnings.filterwarnings("ignore")
import time
import os
import random
import subprocess
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OutputCodeClassifier,OneVsRestClassifier,OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelBinarizer
from scipy import sparse
from sklearn.metrics import *
import json
from pandas import read_csv
from csv import reader
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from libsvm.svmutil import *
from random import sample

def f_statistic(X,Y,num_feats): ######## Univariate Selection using f-statistic or Anova F-value
	feats_remove = []
	test1 = SelectKBest(score_func=f_classif, k=num_feats)
	fit1 = test1.fit(X, Y)
	set_printoptions(precision=3)
	scores = fit1.scores_
	#print("F: ",scores)
	
	max = min = 0
	for score in scores:
		if score > max:
			max = score
		if score < min:
			min = score
	diff = float((max - min)/100)
	per_score = diff*10
	for i in range(len(scores)):
		if scores[i] < per_score:
			feats_remove.append(i)
	return feats_remove

def chi2_statistic(X,Y,num_feats): ######## Univariate Selection using chi-square statistic
	feats_remove = []
	test2 = SelectKBest(score_func=chi2, k=num_feats)
	fit2 = test2.fit(X, Y)
	set_printoptions(precision=3)
	scores = fit2.scores_
	#print("Chi2: ",scores)

	max = min = 0
	for score in scores:
		if score > max:
			max = score
		if score < min:
			min = score
	diff = float((max - min)/100)
	per_score = diff*10
	for i in range(len(scores)):
		if scores[i] < per_score:
			feats_remove.append(i)
	return feats_remove

def Bagged_decision_tree(X,Y,num_feats): ######## Feature Importance using Bagged decision tree: ExtraTreesClassifier
	feats_remove = []
	model = ExtraTreesClassifier(n_estimators=num_feats)
	model.fit(X, Y)
	scores = model.feature_importances_
	#print("BDT: ",scores)

	max = min = 0
	for score in scores:
		if score > max:
			max = score
		if score < min:
			min = score
	diff = float((max - min)/100)
	per_score = diff*10
	for i in range(len(scores)):
		if scores[i] < per_score:
			feats_remove.append(i)
	return feats_remove

def pca(X,Y,num_feats): ######## Principal Component Analysis
	feats_remove = []
	pca = PCA(n_components=num_feats)
	fit = pca.fit(X)
	scores = fit.explained_variance_ratio_
	#print("PCA: ",scores)

	max = min = 0
	for score in scores:
		if score > max:
			max = score
		if score < min:
			min = score
	diff = float((max - min)/100)
	per_score = diff*10
	for i in range(len(scores)):
		if scores[i] < per_score:
			feats_remove.append(i)
	return feats_remove

def dataset_minmax(dataset):
	minmax = list()
	stats = [[min(column), max(column)] for column in zip(*dataset)]
	return stats

def normalize_dataset(dataset, minmax):
	for row in dataset:
		for i in range(len(row)-1):
			if (minmax[i][1] - minmax[i][0]) != 0:
				row[i] = round((row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0]),2)
			else:
				row[i] = 0.0

def balance_data(X,y): # sample equal instances to handle imbalance data
	mini = 0
	alist = [0] * 4
	X_bal = []
	y_bal = []

	labels = [1,2,3,4]
	for i in range(len(X)):
		for k in range(len(labels)):
			if y[i] == labels[k]:
				alist[k] += 1
	mini = min(alist)
	ctr1 = 0
	ctr2 = 0
	ctr3 = 0
	ctr4 = 0	
	for i in range(len(X)):
		if y[i] == 1 and ctr1 < mini:
			X_bal.append(X[i])
			y_bal.append(y[i])
			ctr1 += 1
		elif y[i] == 2 and ctr2 < mini:
			X_bal.append(X[i])
			y_bal.append(y[i])
			ctr2 += 1
		elif y[i] == 3 and ctr3 < mini:
			X_bal.append(X[i])
			y_bal.append(y[i])
			ctr3 += 1
		elif y[i] == 4 and ctr4 < mini:
			X_bal.append(X[i])
			y_bal.append(y[i])
			ctr4 += 1
		else:
			pass
	return X_bal,y_bal

def load_file(filename): # loads the file
	X = []
	y = []
	with open(filename, 'r') as file:
		csv_reader = reader(file)
		for row in csv_reader:
			temp_row = []
			if not row:
				continue
			for i in range(0,len(row)-1):
				temp_row.append(round(float(row[i]),2))
			X.append(temp_row)
			y.append(int(row[len(row)-1]))
	X,y = balance_data(X,y)
	minmax = dataset_minmax(X)
	normalize_dataset(X, minmax)
	return X,y

def feature_selection(X,y): # obtains the optimal features
	feats_remove1 = set(f_statistic(X,y,len(X[0])))
	feats_remove2 = set(chi2_statistic(X,y,len(X[0])))
	feats_remove3 = set(Bagged_decision_tree(X,y,len(X[0])))
	feats_remove4 = set(pca(X,y,len(X[0])))
	feats_remove = list(set.intersection(feats_remove1, feats_remove2, feats_remove3, feats_remove4))

	#feats_remove = set(f_statistic(X,y,len(X[0])))
	print("Optimal features are obtaind by removing features ",feats_remove)
	return feats_remove

def remove_features(X,feats): # remove features to create optimal set
	temp1 = []
	for row in X:
		temp2 = []
		for i in range(0,len(row)):
			if i not in feats:
				temp2.append(row[i])
		temp1.append(temp2)
	return temp1

def classify(X,y): # split in 10-fold for cross validation
	score = [0.0,0.0,0.0,0.0,0.0,0.0,0.0]
	for i in range(0,10):
		temp_scores = classify_fold(X,y,i)
		for j in range(0,len(temp_scores)):
			score[j] += temp_scores[j]
	for j in range(0,len(temp_scores)):
		score[j] = float(score[j]/10)
	return score

def classify_write(X,y): # classifies deficiency using SVM
	clf = svm_train(y, X, '-c 4')
	p_label, p_acc, p_val = svm_predict(y, X, clf)
	return p_label,y

def classify_fold(X,y,i): # classifies deficiency using SVM
	scores = []
	clf = svm_train(y, X, '-c 4')
	p_label, p_acc, p_val = svm_predict(y, X, clf)
	scores.append(accuracy_score(y, p_label))
	scores.append(hamming_loss(y, p_label))
	scores.append(precision_score(y, p_label, average='weighted'))
	scores.append(recall_score(y, p_label, average='weighted'))
	scores.append(f1_score(y, p_label, average='weighted'))
	scores.append(jaccard_score(y, p_label, average='weighted'))
	scores.append(zero_one_loss(y, p_label))
	return scores

def show_res(score): # feature selection, training, and testing
	print("Accuracy: ",score[0])
	print("Hamming loss: ",score[1])
	print("Precision score: ",score[2])
	print("Recall score: ",score[3])
	print("F1 score: ",score[4])
	print("Jaccard score: ",score[5])
	print("Zero one loss: ",score[6])

def predict(): # feature selection, training, and testing
	files2=['Physics1.csv','Chemistry1.csv', 'Mathematics1.csv','Biology1.csv','Science1.csv','Geography1.csv','Economics1.csv','Physics2.csv','Chemistry2.csv', 'Mathematics2.csv','Biology2.csv','Science2.csv','Geography2.csv','Economics2.csv','feature1.csv','feature2.csv']
	#files2=['feature1.csv','feature2.csv']
	path='8_Deficiency/'
	print("Overall Predcition ...")
	for filename in files2:
		print(filename)
		X,y=load_file(path+filename)
		predict_store(X,y)
	
def predict_store(X,y): # feature selection, training, and testing
	feats_remove = feature_selection(X,y)
	X=remove_features(X,feats_remove)
	score = classify(X,y)
	show_res(score)

def write_feature(): # combining the features with gold-standard
	print("Writing feature vector combining gold-standard labels with features .....")
	gold={}
	con = {}
	infile2='GS/GS_Deficiency.txt'
	f=open(infile2,'r')
	ctr = 0
	for line in f:
		ctr+=1
		lpart=line.strip().split(" ")
		key=lpart[0]+' '+lpart[1]+' '+lpart[2].replace("(","").replace(")","")
		val=lpart[3]
		gold[key]=val
		if lpart[2].replace("(","").replace(")","") not in con.keys():
			con[lpart[2].replace("(","").replace(")","")]=ctr
	f.close()

	topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	for t in topics:
		infile1='7_cFeature/'+t+'.json'
		out_dir_name="8_Deficiency/"
		if not os.path.exists(out_dir_name):
			os.makedirs(out_dir_name)
		outfile3='8_Deficiency/'+t+'1.csv'
		outfile4='8_Deficiency/'+t+'2.csv'
		with open(infile1) as json_file:
			data = json.load(json_file)
			for d in data:
				cons = d['topics']
				for i in range(len(cons)):
					key1 = d['id']+" > "+cons[i].replace(" ","_").replace("(","").replace(")","")
					key2 = cons[i].replace(" ","_").replace("(","").replace(")","")
					f3=open(outfile3,'a')
					f3.write(str(d['sf1'])+','+str(d['sf2'])+','+str(d['sf3'])+','+str(d['sf4'])+','+gold[key1]+'\n')
					f3.close()
					f4=open(outfile4,'a')
					f4.write(str(d['sf1'])+','+str(d['sf2'])+','+str(d['sf3'])+','+str(d['sf4'])+','+str(d['cf1'][i])+','+str(d['cf2'][i])+','+str(d['cf3'][i])+','+str(d['cf4'][i])+','+str(d['cf5'][i])+','+str(d['cf6'][i])+','+str(d['cf7'][i])+','+str(d['cf8'][i])+','+str(con[key2])+','+gold[key1]+'\n')
					f4.close()
	
	for t in topics:
		infile1='7_cFeature/'+t+'.json'
		out_dir_name="8_Deficiency/"
		if not os.path.exists(out_dir_name):
			os.makedirs(out_dir_name)
		outfile3='8_Deficiency/feature1.csv'
		outfile4='8_Deficiency/feature2.csv'
		with open(infile1) as json_file:
			data = json.load(json_file)
			for d in data:
				cons = d['topics']
				for i in range(len(cons)):
					key1 = d['id']+" > "+cons[i].replace(" ","_").replace("(","").replace(")","")
					key2 = cons[i].replace(" ","_").replace("(","").replace(")","")
					f3=open(outfile3,'a')
					f3.write(str(d['sf1'])+','+str(d['sf2'])+','+str(d['sf3'])+','+str(d['sf4'])+','+gold[key1]+'\n')
					f3.close()
					f4=open(outfile4,'a')
					f4.write(str(d['sf1'])+','+str(d['sf2'])+','+str(d['sf3'])+','+str(d['sf4'])+','+str(d['cf1'][i])+','+str(d['cf2'][i])+','+str(d['cf3'][i])+','+str(d['cf4'][i])+','+str(d['cf5'][i])+','+str(d['cf6'][i])+','+str(d['cf7'][i])+','+str(d['cf8'][i])+','+gold[key1]+'\n')
					f4.close()

def main(): #main for menu
	start_time = time.time()
	subjects = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	#subjects = ['test']
	write_feature()
	predict()
	print("Exuction time: ", (time.time() - start_time))

if __name__=="__main__":
	main()
