import wikipediaapi
import sys
import os
import json
import random 

def get_inlink(page):
	inlink = []
	for x in page.backlinks:
		if x.startswith("User:"):
			continue
		elif x.startswith("User talk:"):
			continue
		elif x.startswith("Book:"):
			continue
		elif x.startswith("Book talk:"):
			continue
		elif x.startswith("Wikipedia:"):
			continue
		elif x.startswith("Draft:"):
			continue
		elif x.startswith("Portal:"):
			continue
		elif x.startswith("Talk:"):
			continue
		elif x.startswith("Category:"):
			continue
		elif x.startswith("Help:"):
			continue
		elif x.startswith("Template:"):
			continue
		elif x.startswith("Template talk:"):
			continue
		elif x.startswith("Wikipedia talk:"):
			continue
		elif x.startswith("Module talk:"):
			continue
		else:
			inlink.append(x)
	return inlink
	
def get_outlink(page):
	outlink = []
	for x in page.links:
		if x.startswith("User:"):
			continue
		elif x.startswith("User talk:"):
			continue
		elif x.startswith("Book:"):
			continue
		elif x.startswith("Book talk:"):
			continue
		elif x.startswith("Wikipedia:"):
			continue
		elif x.startswith("Draft:"):
			continue
		elif x.startswith("Portal:"):
			continue
		elif x.startswith("Talk:"):
			continue
		elif x.startswith("Category:"):
			continue
		elif x.startswith("Help:"):
			continue
		elif x.startswith("Template:"):
			continue
		elif x.startswith("Template talk:"):
			continue
		elif x.startswith("Wikipedia talk:"):
			continue
		elif x.startswith("Module talk:"):
			continue
		else:
			outlink.append(x)
	return outlink

def getidata(t):
	wiki = wikipediaapi.Wikipedia('en')
	page = wiki.page(t)
	inlinks = get_inlink(page)
	#print(str(len(inlinks)))
	return inlinks

def getodata(t):
	wiki = wikipediaapi.Wikipedia('en')
	page = wiki.page(t)
	outlinks = get_outlink(page)
	#print(str(len(outlinks)))
	return outlinks

def wiki():
	file_in='./4_Concept/uConcepts.txt'
	dir_out_path='./5_Link/'
	if not os.path.exists(dir_out_path):
		os.makedirs(dir_out_path)
	f = open(file_in, "r")
	for line in f:
		print("Processing concept: " + str(line.strip()))
		if not os.path.exists(dir_out_path + line.lower().strip().replace("/","") + "_inlink.txt"):
			ilist = getidata(line.strip().lower())
			f1 = open(dir_out_path + line.lower().strip().replace("/","") + "_inlink.txt", "a")
			for uc in ilist:
				f1.write(uc.lower() + "\n")
			f1.close()
		if not os.path.exists(dir_out_path + line.lower().strip().replace("/","") + "_outlink.txt"):
			olist = getodata(line.strip().lower())
			f2 = open(dir_out_path + line.lower().strip().replace("/","") + "_outlink.txt", "a")
			for uc in olist:
				f2.write(uc.lower() + "\n")
			f2.close()
	f.close()
	print("Collected all Wikipedia links")

def aspect():
	file_in='./4_Concept/uConcepts.txt'
	alist = []
	f = open(file_in, "r")
	for line in f:
		alist.append(line.lower())
	f.close()
	slist = random.sample(alist, 100)
	for s in slist:
		f1 = open('./5_Link/' + s.lower().strip().replace("/","") + "_inlink.txt","r")
		for l1 in f1:
			fr = open('./GS/' + "Related.txt","a")
			fr.write(s.lower().strip().replace("/","")+" > "+l1.strip()+" 1\n")
			fr.close()
			fp = open('./GS/' + "Prerequisite.txt","a")
			fp.write(s.lower().strip().replace("/","")+" > "+l1.strip()+" 1\n")
			fp.close()
		f1.close()
		f2 = open('./5_Link/' + s.lower().strip().replace("/","") + "_outlink.txt","r")
		for l2 in f2:
			fr = open('./GS/' + "Related.txt","a")
			fr.write(s.lower().strip().replace("/","")+" > "+l2.strip()+" 1\n")
			fr.close()
			fd = open('./GS/' + "Dependent.txt","a")
			fd.write(s.lower().strip().replace("/","")+" > "+l2.strip()+" 1\n")
			fd.close()
		f2.close()
	print("Copied 100 random links")

def main(): #main for menu
	wiki()
	aspect()

if __name__ == '__main__':
	main()
