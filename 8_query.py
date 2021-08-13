import warnings
warnings.filterwarnings("ignore")
import time
import re
import os
import json
import random

def select_query(t):
	concepts = []
	f = open('./9_Query/'+t+'_1.txt','r')
	for line in f:
		concepts.append(line.strip())
	f.close()
	tconcepts = random.sample(concepts, 500)
	for tconcept in tconcepts:
		fw = open('./9_Query/'+t+'.txt','a')
		fw.write(tconcept+'\n')
		fw.close()

def generate_query(t,defc):
	dump_file={}
	dump_file['examples']=[]
	dir_in='./6_sFeature/'+t+'.json'
	dir_out='./9_Query/'+t+'.json'
	with open(dir_in) as json_file:
		data = json.load(json_file)
		for d in data:
			idd=d['id']
			conc=d['topics']
			dlist = []
			clist = []
			for c in conc:
				inlist = []
				outlist = []
				f1 = open('5_Link/'+c.replace("/","")+'_inlink.txt', 'r')
				for l1 in f1:
					inlist.append(l1.lower().strip())
				f1.close()
				f2 = open('5_Link/'+c.replace("/","")+'_outlink.txt', 'r')
				for l2 in f2:
					outlist.append(l2.lower().strip())
				f2.close()
				#defc_c = int(defc["test_"+d['id']+"_"+c])
				defc_c = int(defc[d['id']+"_"+c.replace(" ","_")])
				if defc_c == 1:
					clist.append(list(set.union(set.intersection(set(inlist),set(conc)),set.intersection(set(outlist),set(conc)))))
				elif defc_c == 2:
					clist.append(list(set.union(set.intersection(set(inlist),set(conc)),set.intersection(set(outlist),set(conc)))))
				elif defc_c == 3:
					clist.append(list(set.union(set.intersection(set(inlist),set(conc)),set.intersection(set(outlist),set(conc)))))
				else:
					pass
				dlist.append(defc[d['id']+"_"+c.replace(" ","_")])
			dump_file['examples'].append({"id":idd,"topics":conc,"defs":dlist,"conts":clist})
	wfile=os.path.join(dir_out)
	with open(wfile, 'w') as outfile:
		json.dump(dump_file['examples'], outfile, indent=4)

def write_query(t):
	dir_in='./9_Query/'+t+'.json'
	dir_out='./9_Query/'+t+'_1.txt'
	with open(dir_in) as json_file:
		data = json.load(json_file)
		for d in data:
			topics=d['topics']
			defs=d['defs']
			conts=d['conts']
			for i in range(len(topics)):
				for cs in d['conts']:
					query = topics[i].replace(" ","_")+"\t"+topics[i]+" "+str(cs)
					fw = open(dir_out,'a')
					fw.write(d['id']+'\t'+query+'\n')
					fw.close()

def main(): # stores random 2 queries in Grade.json with id,text,query,topics,context
	start_time = time.time()

	subjects = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	#subjects = ['test']
	for s in subjects:
		# create def dict	
		defc = {}
		f=open('8_Deficiency/Deficiency.txt','r')
		for line in f:
			lpart=line.strip().split(" ")
			#defc["test_"+lpart[0]+"_"+lpart[2].replace("_"," ")]=lpart[3]
			defc[lpart[0]+"_"+lpart[2]]=lpart[3]
		f.close()
		generate_query(s,defc)
		# write all queries
		write_query(s)
		# write 2 random queries for each subject
		select_query(s)

	
	print("Exuction time: ", (time.time() - start_time))

if __name__=="__main__":
	main()
