import wikipediaapi
import sys
import os
import json
import random 

def aspect():
	file_in='./4_Concept/uConcepts.txt'
	alist = []
	f = open(file_in, "r")
	for line in f:
		alist.append(line.lower())
	f.close()
	slist = random.sample(alist, 100)
	for s in slist:
		f1 = open('./4_Link/' + s.lower().strip().replace("/","") + "_inlink.txt","r")
		for l1 in f1:
			fr = open('./5_Aspect/' + "Related.txt","a")
			fr.write(s.lower().strip().replace("/","")+" > "+l1.strip()+" 1\n")
			fr.close()
			fp = open('./5_Aspect/' + "Prerequisite.txt","a")
			fp.write(s.lower().strip().replace("/","")+" > "+l1.strip()+" 1\n")
			fp.close()
		f1.close()
		f2 = open('./4_Link/' + s.lower().strip().replace("/","") + "_outlink.txt","r")
		for l2 in f2:
			fr = open('./5_Aspect/' + "Related.txt","a")
			fr.write(s.lower().strip().replace("/","")+" > "+l2.strip()+" 1\n")
			fr.close()
			fd = open('./5_Aspect/' + "Dependent.txt","a")
			fd.write(s.lower().strip().replace("/","")+" > "+l2.strip()+" 1\n")
			fd.close()
		f2.close()
	print("Copied 100 random links")

def main(): #main for menu
	aspect()

if __name__ == '__main__':
	main()
