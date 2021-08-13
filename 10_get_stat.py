import json
import os
import time
import xml.dom.minidom
from sklearn.metrics import cohen_kappa_score

def archive(topics):
	print("\nQA Statistics")
	print("***************")

	for t in topics:
		rt = t
		if t == 'Science':
			rt = 'Physics'
		dir_name='QA/'+rt+'.xml'
		doc = xml.dom.minidom.parse(dir_name)
		rows = doc.getElementsByTagName('row')
		print("Subject: ",t," Number of QA pairs: ",len(rows))

def gs():

	print("\nGS Statistics")
	print("****************")

	print("GS Aspect Statistics")
	print("************************")
	aspect('Concepts')
	aspect('Dependent')
	aspect('Prerequisite')
	aspect('Related')

	print("GS Deficiency Statistics")
	print("************************")
	
	ctr1 = ctr2 = ctr3 = ctr4 = ctr5 = ctr6 = ctr7 = ctr8 = ctr9 = ctr10 = ctr11 = ctr12 = ctr13 = ctr14 = ctr15 = ctr16 =  0
	with open('GS/GS_Deficiency1.txt', 'rt') as f1:
		line1 = f1.readlines()
	f1.close()
	with open('GS/GS_Deficiency2.txt', 'rt') as f2:
		line2 = f2.readlines() 
	f2.close()

	for i in range(len(line1)):
		lpart1 = line1[i].strip().split(" ")
		lpart2 = line2[i].strip().split(" ")
		#print(lpart1[1] + "\t" + lpart2[1])
		if lpart1[3] == '1':
			if lpart2[3] == '1':
				ctr1+=1
			elif lpart2[3] == '2':
				ctr2+=1
			elif lpart2[3] == '3':
				ctr3+=1
			else:
				ctr4+=1
		elif lpart1[3] == '2':
			if lpart2[3] == '1':
				ctr5+=1
			elif lpart2[3] == '2':
				ctr6+=1
			elif lpart2[3] == '3':
				ctr7+=1
			else:
				ctr8+=1
		elif lpart1[3] == '3':
			if lpart2[3] == '1':
				ctr9+=1
			elif lpart2[3] == '2':
				ctr10+=1
			elif lpart2[3] == '3':
				ctr11+=1
			else:
				ctr12+=1
		else:
			if lpart2[3] == '1':
				ctr13+=1
			elif lpart2[3] == '2':
				ctr14+=1
			elif lpart2[3] == '3':
				ctr15+=1
			else:
				ctr16+=1

	print(str(ctr1) + "\t" + str(ctr2) + "\t" + str(ctr3) + "\t" + str(ctr4))
	print(str(ctr5) + "\t" + str(ctr6) + "\t" + str(ctr7) + "\t" + str(ctr8))
	print(str(ctr9) + "\t" + str(ctr10) + "\t" + str(ctr11) + "\t" + str(ctr12))
	print(str(ctr13) + "\t" + str(ctr14) + "\t" + str(ctr15) + "\t" + str(ctr16))
	
	kappa_def()
	
	print("GS Retrieval Statistics")
	print("***********************")
	
	ctr1 = ctr2 = ctr3 = ctr4 = 0
	with open('GS/GS_Retrieval1.txt', 'rt') as f1:
		line1 = f1.readlines()
	f1.close()
	with open('GS/GS_Retrieval2.txt', 'rt') as f2:
		line2 = f2.readlines() 
	f2.close()

	for i in range(len(line1)):
		lpart1 = line1[i].strip().split(" ")
		lpart2 = line2[i].strip().split(" ")
		#print(lpart1[1] + "\t" + lpart2[1])
		if lpart1[3] == '1':
			if lpart2[3] == '1':
				ctr1+=1
			else:
				ctr2+=1
		else:
			if lpart2[3] == '1':
				ctr3+=1
			else:
				ctr4+=1

	print(str(ctr1) + "\t" + str(ctr2))
	print(str(ctr3) + "\t" + str(ctr4))

	kappa_ret()

def corpus(topics):
	print("Corpus Statistics")
	print("*****************")
	ctr = 0 # Number of concepts
	glist = []
	clist = []
	slist = []
	for topicname in topics:
		dir_name='./4_Concept/'+topicname+'/'
		cgrade = ''
		cchapter = ''
		csection = ''
		for dir_section in sorted(os.listdir(dir_name)):
			with open(dir_name + dir_section, 'rt') as f:
				data = json.load(f)
				for d in data:
					#print("dir_section: ",dir_section)
					idpart = dir_section.split("_")
					grade = idpart[1]
					chapter = idpart[2]
					section = idpart[3].replace(".json","")
					if cgrade != grade:
						#print("cgrade: ",cgrade," grade: ",grade)
						#print("glist: ",glist)
						cgrade = grade
						cchapter = ''
						csection = ''
						glist.append(grade)
					if cchapter != chapter:
						#print("cchapter: ",cchapter," chapter: ",chapter)
						#print("clist: ",clist)
						cchapter = chapter
						csection = ''
						clist.append(chapter)
					if csection != section:
						#print("csection: ",csection," section: ",section)
						#print("slist: ",slist)
						csection = section
						slist.append(section)
					ctr += len(d['topics'])
			f.close()
	print("Number of subjects: ", len(topics))
	print("Number of grades: ", len(glist))
	print("Number of chapters: ", len(clist))
	print("Number of sectios: ", len(slist))
	print("Number of concepts: ", ctr)

def kappa_def():

	line1 = []
	line2 = []
	f1 = open('GS/GS_Deficiency1.txt', 'r')
	for l1 in f1:
		line1.append(l1.strip().split(" ")[3])
	f1.close()
	f2 = open('GS/GS_Deficiency2.txt', 'r')
	for l2 in f2:
		line2.append(l2.strip().split(" ")[3])
	f2.close()
	# Overall
	print("Kappa: ",cohen_kappa_score(line1,line2))
	# D1
	ctr1 = ctr2 = ctr3 = ctr4 = 0.0
	for x,y in zip(line1,line2):
		if x == '1':
			if y == '1':
				ctr1 += 1.0
			else:
				ctr2 += 1.0
		else:
			if y == '1':
				ctr3 += 1.0
			else:
				ctr4 += 1.0
	#print(str(ctr1) + "\t" + str(ctr2))
	#print(str(ctr3) + "\t" + str(ctr4))
	total = (ctr1 + ctr2 + ctr3 + ctr4)
	po = (ctr1 + ctr4) / total
	py = ((ctr1 + ctr2) / total) * ((ctr1 + ctr3) / total)
	pn = ((ctr3 + ctr4) / total) * ((ctr2 + ctr4) / total)
	pe = py + pn
	k = round((po - pe) / (1.0 - pe),2)
	print("Kappa D1: ",k)
	line11 = ['2' if x=='3' else x for x in line1]
	line11 = ['2' if x=='4' else x for x in line1]
	line22 = ['2' if x=='3' else x for x in line2]
	line22 = ['2' if x=='4' else x for x in line2]
	#print("Kappa D1: ",cohen_kappa_score(line11,line22))
	# D2
	ctr1 = ctr2 = ctr3 = ctr4 = 0.0
	for x,y in zip(line1,line2):
		if x == '2':
			if y == '2':
				ctr1 += 1.0
			else:
				ctr2 += 1.0
		else:
			if y == '2':
				ctr3 += 1.0
			else:
				ctr4 += 1.0
	#print(str(ctr1) + "\t" + str(ctr2))
	#print(str(ctr3) + "\t" + str(ctr4))
	total = (ctr1 + ctr2 + ctr3 + ctr4)
	po = (ctr1 + ctr4) / total
	py = ((ctr1 + ctr2) / total) * ((ctr1 + ctr3) / total)
	pn = ((ctr3 + ctr4) / total) * ((ctr2 + ctr4) / total)
	pe = py + pn
	k = round((po - pe) / (1.0 - pe),2)
	print("Kappa D2: ",k)
	line11 = ['1' if x=='3' else x for x in line1]
	line11 = ['1' if x=='4' else x for x in line1]
	line22 = ['1' if x=='3' else x for x in line2]
	line22 = ['1' if x=='4' else x for x in line2]
	#print("Kappa D2: ",cohen_kappa_score(line11,line22))
	# D3
	ctr1 = ctr2 = ctr3 = ctr4 = 0.0
	for x,y in zip(line1,line2):
		if x == '3':
			if y == '3':
				ctr1 += 1.0
			else:
				ctr2 += 1.0
		else:
			if y == '3':
				ctr3 += 1.0
			else:
				ctr4 += 1.0
	#print(str(ctr1) + "\t" + str(ctr2))
	#print(str(ctr3) + "\t" + str(ctr4))
	total = (ctr1 + ctr2 + ctr3 + ctr4)
	po = (ctr1 + ctr4) / total
	py = ((ctr1 + ctr2) / total) * ((ctr1 + ctr3) / total)
	pn = ((ctr3 + ctr4) / total) * ((ctr2 + ctr4) / total)
	pe = py + pn
	k = round((po - pe) / (1.0 - pe),2)
	print("Kappa D3: ",k)
	line11 = ['1' if x=='2' else x for x in line1]
	line11 = ['1' if x=='4' else x for x in line1]
	line22 = ['1' if x=='2' else x for x in line2]
	line22 = ['1' if x=='4' else x for x in line2]
	#print("Kappa D3: ",cohen_kappa_score(line11,line22))
	# D4
	ctr1 = ctr2 = ctr3 = ctr4 = 0.0
	for x,y in zip(line1,line2):
		if x == '4':
			if y == '4':
				ctr1 += 1.0
			else:
				ctr2 += 1.0
		else:
			if y == '4':
				ctr3 += 1.0
			else:
				ctr4 += 1.0
	#print(str(ctr1) + "\t" + str(ctr2))
	#print(str(ctr3) + "\t" + str(ctr4))
	total = (ctr1 + ctr2 + ctr3 + ctr4)
	po = (ctr1 + ctr4) / total
	py = ((ctr1 + ctr2) / total) * ((ctr1 + ctr3) / total)
	pn = ((ctr3 + ctr4) / total) * ((ctr2 + ctr4) / total)
	pe = py + pn
	k = round((po - pe) / (1.0 - pe),2)
	print("Kappa D4: ",k)
	line11 = ['1' if x=='2' else x for x in line1]
	line11 = ['1' if x=='3' else x for x in line1]
	line22 = ['1' if x=='2' else x for x in line2]
	line22 = ['1' if x=='3' else x for x in line2]
	#print("Kappa D4: ",cohen_kappa_score(line11,line22))

def kappa_ret():

	line1 = []
	line2 = []
	f1 = open('GS/GS_Retrieval1.txt', 'r')
	for l1 in f1:
		line1.append(l1.strip().split(" ")[3])
	f1.close()
	f2 = open('GS/GS_Retrieval2.txt', 'r')
	for l2 in f2:
		line2.append(l2.strip().split(" ")[3])
	f2.close()

	ctr1 = ctr2 = ctr3 = ctr4 = 0.0
	for x,y in zip(line1,line2):
		if x == '1':
			if y == '1':
				ctr1 += 1.0
			else:
				ctr2 += 1.0
		else:
			if y == '1':
				ctr3 += 1.0
			else:
				ctr4 += 1.0
	#print(str(ctr1) + "\t" + str(ctr2))
	#print(str(ctr3) + "\t" + str(ctr4))
	total = (ctr1 + ctr2 + ctr3 + ctr4)
	po = (ctr1 + ctr4) / total
	py = ((ctr1 + ctr2) / total) * ((ctr1 + ctr3) / total)
	pn = ((ctr3 + ctr4) / total) * ((ctr2 + ctr4) / total)
	pe = py + pn
	k = round((po - pe) / (1.0 - pe),2)
	print("Kappa: ",k)
	#print("Kappa: ",cohen_kappa_score(line1,line2))

def aspect(t):
	i=0
	ctr=0
	f1 = open('GS/'+t+'.txt','r')
	for line1 in f1:
		i+=1
		lpart = line1.strip().split(" ")
		if lpart[-1] == '1':
			ctr+=1
	f1.close()
	perf1=round(float(ctr)/float(i),2)
	print(t+' Accuracy:\t'+str(perf1))

def main(): #main for menu
	topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	corpus(topics)
	archive(topics)
	gs()

if __name__=="__main__":
	main()
