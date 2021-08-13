import warnings
warnings.filterwarnings("ignore")
import time
import os
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

def test(filename):
	y, x = svm_read_problem(filename)
	y_train = y[:200]
	y_test = y[200:]
	x_train = x[:200]
	x_test = x[200:]
	clf = svm_train(y_train, x_train, '-c 4')
	p_label, p_acc, p_val = svm_predict(y_test, x_test, clf)
	print("*****************************************")
	print("Accuracy: ",accuracy_score(y_test, p_label)) # accuracy
	print("Hamming loss: ",hamming_loss(y_test, p_label)) # hamming loss
	print("Precision score: ",precision_score(y_test, p_label, average='weighted')) # precision score
	print("Recall score: ",recall_score(y_test, p_label, average='weighted')) # recall score
	print("F1 score: ",f1_score(y_test, p_label, average='weighted')) # f1 score
	print("jaccard_score: ",jaccard_score(y_test, p_label, average='weighted')) # jaccard_score
	print("zero_one_loss: ",zero_one_loss(y_test, p_label)) # zero_one_loss

def test1():
	y, X = svm_read_problem('feature.txt')
	y_dense = LabelBinarizer().fit_transform(y)
	y_sparse = sparse.csr_matrix(y_dense)
	clf = OutputCodeClassifier(LinearSVC(random_state=0),code_size=2, random_state=0)
	pred2 = clf.fit(X, y_sparse).predict(X)
	print("Accuracy: ",metrics.accuracy_score(y_sparse.todense(), pred2.todense()))

#For example the ANOVA F-value method is appropriate for numerical inputs and categorical data, as we see in the Pima dataset.
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

def balance_data(X,y,z): # sample equal instances to handle imbalance data
	mini = 0
	alist = [0] * 28
	X_bal = []
	y_bal = []
	z_bal = []
	subjects = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	labels = [1,2,3,4]
	for i in range(len(X)):
		for j in range(len(subjects)):
			for k in range(len(labels)):
				if y[i] == labels[k] and z[i].startswith(subjects[j]):
					alist[4*j+k] += 1
	mini = min(alist)
	for i in range(len(alist)):
		for k in range(len(labels)):
			if (i+1)%(k+1) == 1:
				alist[i] = 1 * mini
			if (i+1)%(k+1) == 2:
				alist[i] = 1 * mini
			if (i+1)%(k+1) == 3:
				alist[i] = 1 * mini
			if (i+1)%(k+1) == 4:
				alist[i] = 1 * mini
			else:
				pass
	#alist = [mini] * 28
	for i in range(len(X)):
		for j in range(len(subjects)):
			for k in range(len(labels)):
				if y[i] == labels[k] and z[i].startswith(subjects[j]) and alist[4*j+k] > 0:
					alist[4*j+k] -= 1
					X_bal.append(X[i])
					y_bal.append(y[i])
					z_bal.append(z[i])
	return X_bal,y_bal,z_bal

def load_file(filename): # loads the file
	X = []
	y = []
	z =[]
	subjects = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	for t in subjects:
		infile1='7_cFeature/'+t+'.json'
		with open(infile1) as json_file:
			data = json.load(json_file)
			for d in data:
				cons = d['topics']
				for i in range(len(cons)):
					key = d['id']+" > "+cons[i].replace(" ","_").replace("(","").replace(")","")
					z.append(key)
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
	X,y,z = balance_data(X,y,z)
	minmax = dataset_minmax(X)
	normalize_dataset(X, minmax)
	return X,y,z

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
	files2=['feature1.csv','feature2.csv']
	path='8_Deficiency/'
	print("Overall Predcition ...")
	for filename in files2:
		print(filename)
		X,y,z=load_file(path+filename)
		predict_store(X,y,z)
	
def predict_store(X,y,z): # feature selection, training, and testing
	feats_remove = feature_selection(X,y)
	X=remove_features(X,feats_remove)
	y_pred,y_test = classify_write(X,y)
	score = classify(X,y)
	show_res(score)
	prf(y_pred,y_test)
	class_def_sub(y_pred,y_test,z)

def club_annotation(): # writing individual annotations in a single file
	infile1='GS/GS_Deficiency1.txt'
	infile2='GS/GS_Deficiency2.txt'
	outfile='GS/GS_Deficiency.txt'
		
	f1=open(infile1,'r')
	f2=open(infile2,'r')
	for x, y in zip(f1, f2):
		if x == y:
			f3=open(outfile,'a')
			f3.write(y.strip()+'\n')
			f3.close()
		else:
			f3=open(outfile,'a')
			f3.write(y.strip()+'\n')
			f3.close()
	f1.close()
	f2.close()
	print("Merged individual annotation files .....")

def write_feature(topics): # combining the features with gold-standard
	print("Writing feature vector combining gold-standard labels with features .....")
	gold={}
	infile2='GS/GS_Deficiency.txt'
	f=open(infile2,'r')
	ctr = 0
	for line in f:
		ctr+=1
		lpart=line.strip().split(" ")
		key=lpart[0]+' '+lpart[1]+' '+lpart[2].replace("(","").replace(")","")
		val=lpart[3]
		gold[key]=val
	f.close()

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

def generate_annotation(topics): # combines cfeature and sfeature in feature.txt
	for t in topics:
		infile='7_cFeature/'+t+'.json'
		outfile1='GS/GS_Deficiency1.txt'
		outfile2='GS/GS_Deficiency2.txt'
		with open(infile) as json_file:
			data = json.load(json_file)
			for d in data:
				cons = d['topics']
				for i in range(len(cons)):
					f5=open(outfile1,'a')
					f5.write(d['id']+" > "+cons[i].replace(" ","_")+' 4\n')
					f5.close()
					f6=open(outfile2,'a')
					f6.write(d['id']+" > "+cons[i].replace(" ","_")+' 4\n')
					f6.close()
	print("Stored annotation files for deficiency .....")

def class_def_sub(y_pred,y_test,z):
	slist = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	for s in slist:
		print(s)
		glist = []
		dlist = []
		for i in range(len(z)):
			lpart = z[i].split(" ")
			if lpart[0].startswith(s):
				dlist.append(y_pred[i])
				glist.append(y_test[i])
		prf(dlist,glist)

def prf(dlist,glist):
	ctr1=0
	ctr2=0
	ctr3=0
	ctr4=0
	ctr5=0
	ctr6=0
	ctr7=0
	ctr8=0
	ctr9=0
	ctr10=0
	ctr11=0
	ctr12=0

	for i in range(len(glist)):
		if glist[i] == 1:
			if dlist[i] == 1:
				ctr1+=1
		if glist[i] == 1:
			ctr2+=1
		if dlist[i] == 1:
			ctr3+=1

		if glist[i] == 2:
			if dlist[i] == 2:
				ctr4+=1
		if glist[i] == 2:
			ctr5+=1
		if dlist[i] == 2:
			ctr6+=1
		
		if glist[i] == 3:
			if dlist[i] == 3:
				ctr7+=1
		if glist[i] == 3:
			ctr8+=1
		if dlist[i] == 3:
			ctr9+=1
			
		if glist[i] == 4:
			if dlist[i] == 4:
				ctr10+=1
		if glist[i] == 4:
			ctr11+=1
		if dlist[i] == 4:
			ctr12+=1

	if ctr3 == 0:
		d1_p = 0.0
	else:
		d1_p = round(float(ctr1/ctr3),2)
	if ctr2 == 0:
		d1_r = 0.0
	else:
		d1_r = round(float(ctr1/ctr2),2)
	if (d1_p + d1_r) == 0:
		d1_f = 0.0
	else:
		d1_f = round(((2 * d1_p * d1_r) / (d1_p + d1_r)),2)

	if ctr6 == 0:
		d2_p = 0.0
	else:
		d2_p = round(float(ctr4/ctr6),2)
	if ctr5 == 0:
		d2_r = 0.0
	else:
		d2_r = round(float(ctr4/ctr5),2)
	if (d2_p + d2_r) == 0:
		d2_f = 0.0
	else:
		d2_f = round(((2 * d2_p * d2_r) / (d2_p + d2_r)),2)

	if ctr9 == 0:
		d3_p = 0.0
	else:
		d3_p = round(float(ctr7/ctr9),2)
	if ctr8 == 0:
		d3_r = 0.0
	else:
		d3_r = round(float(ctr7/ctr8),2)
	if (d3_p + d3_r) == 0:
		d3_f = 0.0
	else:
		d3_f = round(((2 * d3_p * d3_r) / (d3_p + d3_r)),2)
	
	if ctr12 == 0:
		d4_p = 0.0
	else:
		d4_p = round(float(ctr10/ctr12),2)
	if ctr11 == 0:
		d4_r = 0.0
	else:
		d4_r = round(float(ctr10/ctr11),2)
	if (d4_p + d4_r) == 0:
		d4_f = 0.0
	else:
		d4_f = round(((2 * d4_p * d4_r) / (d4_p + d4_r)),2)

	print(str(ctr1)+"/"+str(ctr2)+"/"+str(ctr3)+"/"+str(ctr4)+"/"+str(ctr5)+"/"+str(ctr6)+"/"+str(ctr7)+"/"+str(ctr8)+"/"+str(ctr9)+"/"+str(ctr10)+"/"+str(ctr11)+"/"+str(ctr12) )

	print(str(d1_p)+" : "+str(d1_r)+" : "+str(d1_f))
	print(str(d2_p)+" : "+str(d2_r)+" : "+str(d2_f))
	print(str(d3_p)+" : "+str(d3_r)+" : "+str(d3_f))
	print(str(d4_p)+" : "+str(d4_r)+" : "+str(d4_f))

def main(): #main for menu
	start_time = time.time()
	subjects = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	#subjects = ['test']
	generate_annotation(subjects)
	club_annotation()
	write_feature(subjects)
	predict()
	print("Exuction time: ", (time.time() - start_time))

if __name__=="__main__":
	main()
