import warnings
warnings.filterwarnings("ignore")
import time
import os
import random
import json
import re
from yattag import Doc, indent

def process(mystring):
	#re.sub('^[a-zA-Z0-9_-]+$',"",mystring)
	return re.sub('[^a-zA-Z0-9 ]','',mystring)

def generate_augment(topic): # combining the features with gold-standard

	aug = {}
	infile1='10_Retrieval/AUG.txt'
	f1=open(infile1,'r')
	for line in f1:
		lpart=line.strip().split(" ")
		if lpart[3] == '1' and lpart[0].startswith(topic):
			llpart=lpart[2].split("_")
			id = "_".join(llpart[0:-1])
			augm = llpart[-1]
			if id not in aug.keys():
				aug[id] = augm
	f1.close()
	#print(len(aug.keys()))
	absent = []
	for key in aug.keys():
		qid = aug[key]
		infile2='QA/QA.txt'
		f2=open(infile2,'r')
		for line in f2:
			lpart=line.strip().split("\t")
			if qid == lpart[0]:
				absent.append(qid)
		f2.close()
	#print(len(absent))

	infile='7_cFeature/'+topic+'.json'
	with open(infile) as json_file:
		data = json.load(json_file)
		for d in data:
			idpart = d['id'].split("_")
			del idpart[-1]
			idd = "_".join(idpart)
			ptext = process(d['text'])
			mentions = d['mentions']
			mentions.sort(key = len)
			for m in mentions:
				ptext = ptext.replace(m,"#"+m+"#",1)
			fw = open('11_Augmentation/'+idd+'.txt','a')
			fw.write(ptext+"\n")
			fw.close()

def main(): #main for menu
	start_time = time.time()
	subjects = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	for s in subjects:
		print(s)
		generate_augment(s)
	print("Exuction time: ", (time.time() - start_time))

if __name__=="__main__":
	main()
