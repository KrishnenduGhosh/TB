from __future__ import absolute_import, division, print_function, unicode_literals
import sys
import tagme
import os
import json

dump_file={}
dump_file['examples']=[]

def main():
	topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	#topics = ['test']
	for s in topics:
		extract(s)
		get_uconcepts(s)
	print("Key-concepts extracted ...........")
	os.system("sort 4_Concept/Concepts.txt | uniq > 4_Concept/uConcepts.txt")
	grade()
	get_first()
	print("Concepts extracted for different grade-levels ...........")

def get_uconcepts(t):
	tlist = []
	dir_in='./4_Concept/'
	for jfile in sorted(os.listdir(dir_in+t)):
		with open(dir_in + t + "/" + jfile) as json_file:
			data = json.load(json_file)
			for d in data:
				for tt in d['topics']:
					if tt not in tlist:
						tlist.append(tt.strip())
	f = open(dir_in + "Concepts.txt", "a")
	for uc in tlist:
		f.write(uc + "\n")
	f.close()

def grade():
	topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
	#topics = ['test']
	for t in topics:
		#print(t)
		dir_in='./4_Concept/'+t+'/'
		for filename in sorted(os.listdir(dir_in)):
			with open(dir_in+filename) as json_file:
				data = json.load(json_file)
				for d in data:
					idpart=filename.split("_") #Physics_11_1_0.json
					dir_out='./4_Concept/'+str(idpart[1])+'_1.txt'
					con=d['topics']
					for c in con:
						fr = open(dir_out,'a')
						fr.write(c.lower().strip().replace("/","")+"\n")
						fr.close()

def get_first():
	grades=['6','7','8','9','10','11','12']
	#grades=['11']
	cgrades = {}
	for g in grades:
		cgrade = []
		dir_in='./4_Concept/'+g+'_1.txt'
		f1 = open(dir_in,'r')
		for l in f1:
			cgrade.append(l.strip())
		f1.close()
		cgrade = set(cgrade)
		cgrades[g]=cgrade
	for i in range(6,12):
		for j in range(7,13):
			if j > i:
				con = [value for value in cgrades[str(i)] if value in cgrades[str(j)]]
				con = set(con)
				for c in con:
					cgrades[str(j)].remove(c)
				udict = {str(j): cgrades[str(j)]}
				cgrades.update(udict)
	for g in grades:
		dir_in='./4_Concept/'+g+'.txt'
		f = open(dir_in,'w')
		f.close()
		for c in cgrades[g]:
			f = open(dir_in,'a')
			f.write(c+"\n")
			f.close()

def extract(t):
	dump_file['examples']=[]
	dir_name="3_Section/"+t
	out_dir_name="4_Concept/"
	if not os.path.exists(out_dir_name):
		os.makedirs(out_dir_name)
	#print(t)
	if not os.path.exists(out_dir_name+t):
		os.makedirs(out_dir_name+t)
	ctr = 0
	for lecture in sorted(os.listdir(dir_name)):
		#print(lecture)
		for filename in sorted(os.listdir(os.path.join(dir_name, lecture))):
			dump_file['examples']=[]
			file_path="./"+dir_name+"/"+lecture+"/"+filename
			file_opath="./"+out_dir_name+t+"/"
			with open(file_path, 'r') as myfile:
				data=myfile.read().encode().decode('utf-8').replace('\n', '').lower()
				# print(data)
				# print("Annotating text: ", data)
				resp = tagme.annotate(data)
				topics_list=[]
				mention_list=[]
				location_list=[]
				dir_part = dir_name.split("/")
				id_no=dir_part[1]+"_"+lecture.split('.')[0]+"_"+filename.split('.')[0]
				#print("id_no: " + id_no)
				if (resp!=None and resp.get_annotations(0.4)!=None):
					for ann in resp.get_annotations(0.4):
						if ann.mention not in mention_list:
							topics_list.append(ann.entity_title.lower())
							mention_list.append(ann.mention.lower())
				#print(topics_list)
				dump_file['examples'].append({"text":data,"topics":topics_list,"mentions":mention_list})
			wfile_name=file_opath+id_no+".json"
			#print("wfile_name: ",wfile_name)
			with open(wfile_name, 'w') as outfile:  
				json.dump(dump_file['examples'], outfile, indent=4)

if __name__ == "__main__":
	tagme.GCUBE_TOKEN = "0b4eed68-e456-4488-a5a6-7a608ea7e32b-843339462"
	main()
