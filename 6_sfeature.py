import os
import word2vec
from math import log
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import nltk
from lib.textsplit.tools import get_penalty, get_segments
from lib.textsplit.algorithm import split_optimal, split_greedy, get_total
import re
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.qparser import QueryParser
from whoosh import scoring
import time
import json
import collections
import re
import corenlp

fam = {'a','bee','built','cook','cost','about','been','burn','cool','above','before','busy','corn','across','began','but','corner','act','afraid','begin','butter','after','behind','buy','could','afternoon','being','by','count','again','believe','cake','country','against','bell','call','course','ago','belong','came','cover','air','beside','can','cow','all','best','cap','cried','almost','better','captain','cross','alone','between','car','crowd','along','big','care','crown','already','bill','careful','cry','also','bird','carry','cup','always','bit','case','cut','am','black','cat','dance','American','bless','catch','dark','an','blind','cause','day','and','blood','cent','dead','animal','blow','enter','dear','another','blue','chair','deep','answer','board','chance','did','any','boat','change','die','anything','body','chief','different','apple','bone','child','dinner','are','book','children','do','arm','born','choose','doctor','around','both','Christmas','does','as','bottom','church','dog','ask','bow','circle','done','at','box','city','door','away','boy','class','double','baby','branch','clean','down','back','brave','clear','draw','bad','bread','clock','dream','bag','break','close','dress','ball','breakfast','cloth','drink','band','bridge','clothes','drive','bank','bright','cloud','drop','basket','bring','coal','dry','be','broken','coat','dust','bear','brother','cold','each','beat','brought','color','ear','beautiful','brown','come','early','because','build','coming','earth','bed','building','company','east','easy','fit','half','just','eat','five','hall','keep','edge','fix','hand','kept','egg','floor','hang','kill','eight','flower','happy','kind','either','fly','hard','king','eleven','follow','has','kiss','else','food','hat','knee','end','foot','have','knew','England','for','he','know','English','forget','head','lady','enough','found','hear','laid','even','four','heard','lake','evening','fourth','heart','land','ever','fresh','heavy','large','every','friend','help','last','everything','from','her','late','except','front','here','laugh','except','fruit','herself','lay','eye','full','hide','lead','face','game','high','learn','fair','garden','hill','leave','fall','gate','him','left','family','gave','himself','leg','fancy','get','his','lesson','far','gift','hold','let','farm','girl','hole','letter','farmer','give','home','lift','fast','glad','hope','light','fat','glass','horse','like','father','go','hot','line','feed','God','house','lion','feel','going','how','lips','feet','gold','hundred','listen','fell','golden','hunt','lit','fellow','gone','hurry','little','felt','good','kurt','live','fence','got','I','load','few','grain','ice','long','field','grass','if','look','fight','gray','in','lost','fill','great','Indian','lot','find','green','instead','loud','fine','grew','into','love','soldier','tear','uncle','wind','some','tell','under','window','something','ten','until','wing','sometime','than','up','winter','song','thank','upon','wish','soon','that','us','with','sound','the','use','without','south','their','valley','woman','space','them','very','wonder','speak','then','visit','wood','spot','there','wait','word','spread','these','walk','work','spring','they','wall','world','square','thick','want','would','stand','thin','war','write','star','thing','warn','wrong','start','think','was','yard','station','this','wash','year','stay','those','waste','yellow','step','though','watch','yes','stick','thought','water','yesterday','mile','number','ran','several','milk','oak','rather','shake','mill','ocean','reach','shall','mind','of','read','shape','mine','on','ready','she','minute','once','real','sheep','miss','one','reason','shine','money','only','red','ship','month','open','remember','shoe','moon','or','rest','shop','more','other','rich','short','morning','our','ride','should','most','out','right','shoulder','mother','outside','ring','show','mountain','over','river','shut','mouth','own','road','sick','move','page','rock','side','Mr.','paint','roll','sign','finger','around','iron','low','finish','grow','is','made','fire','guess','it','mail','first','had','its','make','fish','hair','jump','man','many','nice','poor','season','march','night','post','seat','mark','nine','pound','second','market','no','present','see','may','noise','press','seed','me','none','pretty','seem','mean','noon','pull','seen','measure','nor','put','self','meat','north','quarter','sell','meet','nose','queen','send','men','not','quick','sent','met','note','quiet','serve','middle','nothing','quite','set','might','now','race','seven'}

def f1(text): #Average sentence length
	asl = 0
	ctr = 0
#	print("text: ",text)
	sents = re.split('\. |\? |\! |\.|\?|\!',text)
	for s in sents:
		if s != '':
			ctr+=1
#			print("sentence: ",s)
			words = s.split(" ")
			words = re.split(' |\, |\; |\: |\,|\;|\:',s)
			ctr2 = 0
			for w in words:
				if w != '':
					asl += 1
					ctr2 += 1
#			print("sentence len : ",ctr2)
	if ctr > 1:
		asl = asl / ctr
#	print("sent len: ",ctr)
#	print("asl: ",asl)
	return asl

def f2(text): #Average word length
	awl = 0
	ctr = 0
#	print("text: ",text)
	sents = re.split('\. |\? |\! |\.|\?|\!',text)
#	print("sent len: ",(len(sents)-1))
	for s in sents:
		if s != '':
#			print("sentence: ",s)
			words = re.split(' |\, |\; |\: |\,|\;|\:',s)
			for w in words:
				if w != '':
					awl += len(list(w))
					ctr += 1
#					print("len(letters): ",len(list(w)))
	if len(sents) > 1:
		awl = awl / ctr
#	print("word len: ",ctr)
#	print("awl: ",awl)
	return awl

def f3(text): #Average word familiarity
	awf = 0
	tw = 0
	sents = re.split('\. |\? |\! |\.|\?|\!',text)
	for s in sents:
		if s != '':
			words = s.split(" ")
			for w in words:
				if w != '':
					tw += 1
					if w in fam:
						awf += 1
	if tw > 1:
		awf = awf / tw
	return awf

def f4(text, topics): #Dispersion between concepts
	inlinks=[]
	outlinks=[]
	for t in topics:
		inlink=[]
		outlink=[]
		dir_in='./4_Link/'+t.replace("/","")
		f1 = open(dir_in+'_inlink.txt', 'r')
		for l1 in f1:
			inlink.append(l1.lower().strip())
		f1.close()
		f2 = open(dir_in+'_outlink.txt', 'r')
		for l2 in f2:
			outlink.append(l2.lower().strip())
		f2.close()
#		print("len(inlink): ",len(inlink))
#		print("len(outlink): ",len(outlink))
		inlinks.append(inlink)
		outlinks.append(outlink)
	link = 0
	disp = 0
#	print("len(inlinks): ",len(inlinks))
#	print("len(outlinks): ",len(outlinks))
	for i in range(len(topics)):
		for j in range(len(topics)):
			if i != j:
#				print("topics[i]: ",topics[i])
#				print("topics[j]: ",topics[j])
#				print("inlinks[i]: ",inlinks[i])
#				print("outlinks[i]: ",outlinks[i])
#				print("inlinks[j]: ",inlinks[j])
#				print("outlinks[j]: ",outlinks[j])
				if ( (topics[j] in inlinks[i]) or (topics[j] in outlinks[i]) or (topics[i] in inlinks[j]) or (topics[i] in outlinks[j]) ):
					link += 1
#	print("link: ",link)
#	print("len(topics): ",len(topics))
	if len(topics) > 1:
		disp = 1 - ( (link) / (len(topics) * (len(topics)-1) ))
	return disp

def run(t):
	dump_file={}
	dump_file['examples']=[]
	dir_in='./4_Concept/'+t+'/'
	dir_out='./6_sFeature/'+t+'/'
	if not os.path.exists(dir_out):
		os.makedirs(dir_out)
	for filename in sorted(os.listdir(dir_in)):
		dump_file['examples']=[]
		with open(dir_in+filename) as json_file:
			data = json.load(json_file)
			for d in data:
#				print("filename: ",filename)
				cf1 = f1(d['text'])
				cf2 = f2(d['text'])
				cf3 = f3(d['text'])
				cf4 = f4(d['text'],d['topics'])
				dump_file['examples'].append({"cf1":cf1,"cf2":cf2,"cf3":cf3,"cf4":cf4})
		filename=os.path.join(dir_out+filename)
		with open(filename, 'w') as outfile:
			json.dump(dump_file['examples'], outfile, indent=4)

def main(): #main for menu
	start_time = time.time()
	topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	topics = ['test']
	for t in topics:
		print(t)
		run(t)
#	print("cf1: ",f1(text))
#	print("cf2: ",f2(text))
#	print("cf3: ",f3(text))
#	print("cf4: ",f4(text,cons))
	print("Exuction time: ", (time.time() - start_time))

if __name__ == '__main__':
	main()

