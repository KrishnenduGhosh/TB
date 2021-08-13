from math import log
import numpy as np
import re
import time
import json
import collections
import re

fam = {'a','bee','built','cook','cost','about','been','burn','cool','above','before','busy','corn','across','began','but','corner','act','afraid','begin','butter','after','behind','buy','could','afternoon','being','by','count','again','believe','cake','country','against','bell','call','course','ago','belong','came','cover','air','beside','can','cow','all','best','cap','cried','almost','better','captain','cross','alone','between','car','crowd','along','big','care','crown','already','bill','careful','cry','also','bird','carry','cup','always','bit','case','cut','am','black','cat','dance','American','bless','catch','dark','an','blind','cause','day','and','blood','cent','dead','animal','blow','enter','dear','another','blue','chair','deep','answer','board','chance','did','any','boat','change','die','anything','body','chief','different','apple','bone','child','dinner','are','book','children','do','arm','born','choose','doctor','around','both','Christmas','does','as','bottom','church','dog','ask','bow','circle','done','at','box','city','door','away','boy','class','double','baby','branch','clean','down','back','brave','clear','draw','bad','bread','clock','dream','bag','break','close','dress','ball','breakfast','cloth','drink','band','bridge','clothes','drive','bank','bright','cloud','drop','basket','bring','coal','dry','be','broken','coat','dust','bear','brother','cold','each','beat','brought','color','ear','beautiful','brown','come','early','because','build','coming','earth','bed','building','company','east','easy','fit','half','just','eat','five','hall','keep','edge','fix','hand','kept','egg','floor','hang','kill','eight','flower','happy','kind','either','fly','hard','king','eleven','follow','has','kiss','else','food','hat','knee','end','foot','have','knew','England','for','he','know','English','forget','head','lady','enough','found','hear','laid','even','four','heard','lake','evening','fourth','heart','land','ever','fresh','heavy','large','every','friend','help','last','everything','from','her','late','except','front','here','laugh','except','fruit','herself','lay','eye','full','hide','lead','face','game','high','learn','fair','garden','hill','leave','fall','gate','him','left','family','gave','himself','leg','fancy','get','his','lesson','far','gift','hold','let','farm','girl','hole','letter','farmer','give','home','lift','fast','glad','hope','light','fat','glass','horse','like','father','go','hot','line','feed','God','house','lion','feel','going','how','lips','feet','gold','hundred','listen','fell','golden','hunt','lit','fellow','gone','hurry','little','felt','good','kurt','live','fence','got','I','load','few','grain','ice','long','field','grass','if','look','fight','gray','in','lost','fill','great','Indian','lot','find','green','instead','loud','fine','grew','into','love','soldier','tear','uncle','wind','some','tell','under','window','something','ten','until','wing','sometime','than','up','winter','song','thank','upon','wish','soon','that','us','with','sound','the','use','without','south','their','valley','woman','space','them','very','wonder','speak','then','visit','wood','spot','there','wait','word','spread','these','walk','work','spring','they','wall','world','square','thick','want','would','stand','thin','war','write','star','thing','warn','wrong','start','think','was','yard','station','this','wash','year','stay','those','waste','yellow','step','though','watch','yes','stick','thought','water','yesterday','mile','number','ran','several','milk','oak','rather','shake','mill','ocean','reach','shall','mind','of','read','shape','mine','on','ready','she','minute','once','real','sheep','miss','one','reason','shine','money','only','red','ship','month','open','remember','shoe','moon','or','rest','shop','more','other','rich','short','morning','our','ride','should','most','out','right','shoulder','mother','outside','ring','show','mountain','over','river','shut','mouth','own','road','sick','move','page','rock','side','Mr.','paint','roll','sign','finger','around','iron','low','finish','grow','is','made','fire','guess','it','mail','first','had','its','make','fish','hair','jump','man','many','nice','poor','season','march','night','post','seat','mark','nine','pound','second','market','no','present','see','may','noise','press','seed','me','none','pretty','seem','mean','noon','pull','seen','measure','nor','put','self','meat','north','quarter','sell','meet','nose','queen','send','men','not','quick','sent','met','note','quiet','serve','middle','nothing','quite','set','might','now','race','seven'}

def f1(con,topics): # Number of related concepts absent in section
  rca = 0
  related=[]
  dir_in='./5_Link/'+con.replace("/","")
  f1 = open(dir_in+'_inlink.txt', 'r')
  for l1 in f1:
    related.append(l1.lower().strip())
  f1.close()
  f2 = open(dir_in+'_outlink.txt', 'r')
  for l2 in f2:
    related.append(l2.lower().strip())
  f2.close()
  rcalist = [value for value in related if value not in topics]
  rca = len(rcalist)
  return rca

def f2(con,topics): # Number of prerequisite concepts absent in section
  pca = 0
  prerequisite=[]
  dir_in='./5_Link/'+con.replace("/","")
  f1 = open(dir_in+'_inlink.txt', 'r')
  for l1 in f1:
    prerequisite.append(l1.lower().strip())
  f1.close()
  pcalist = [value for value in prerequisite if value not in topics]
  pca = len(pcalist)
  return pca

def f3(con,topics): # Number of dependent concepts absent in section
  dca = 0
  dependent=[]
  dir_in='./5_Link/'+con.replace("/","")
  f1 = open(dir_in+'_outlink.txt', 'r')
  for l1 in f1:
    dependent.append(l1.lower().strip())
  f1.close()
  dcalist = [value for value in dependent if value not in topics]
  dca = len(dcalist)
  return dca

def f4(con,topics): # Concept significance score
  sc = 0
  related=[]
  dir_in='./5_Link/'+con.replace("/","")
  f1 = open(dir_in+'_inlink.txt', 'r')
  for l1 in f1:
    related.append(l1.lower().strip())
  f1.close()
  f2 = open(dir_in+'_outlink.txt', 'r')
  for l2 in f2:
    related.append(l2.lower().strip())
  f2.close()
  rsc = [value for value in related if value in topics]
  fcs = 0
  for t in topics:
    if t == con:
      fcs += 1
  if len(rsc) > 1:
    sc = fcs * len(rsc)
  return sc

def f5(t,con,idd): # Number of key-sections
  ks = 0
  text_dict = {}
  idd = idd.split("_")[0]
  infile='./6_sFeature/'+t+'.json'
  with open(infile) as json_file:
    data = json.load(json_file)
    for d in data:
      if d['id'].startswith(idd):
        text_dict[d['id']]=d['sf5']
  for key in text_dict:
    if con in text_dict[key]:
      ks += 1
  return ks

def dist(s,s_k): # distance between sections
  return abs(int(s_k.split("_")[2])-int(s.split("_")[2]))

def f6(t,con,idd): # average section distance between related concepts
  asd_r = 0.0
  ctr = 0
  
  related=[]
  dir_in='./5_Link/'+con.replace("/","")
  f1 = open(dir_in+'_inlink.txt', 'r')
  for l1 in f1:
    related.append(l1.lower().strip())
  f1.close()
  f2 = open(dir_in+'_outlink.txt', 'r')
  for l2 in f2:
    related.append(l2.lower().strip())
  f2.close()
  
  gidd = idd.split("_")[0]+"_"+idd.split("_")[1]
  infile='./6_sFeature/'+t+'.json'
  with open(infile) as json_file:
    data = json.load(json_file)
    for d in data:
      if d['id'].startswith(gidd):
        for tt in d['topics']:
          if tt in related:
            asd_r += dist(d['id'],idd)
            ctr += 1
  if ctr >= 1:
    asd_r = asd_r / ctr
  return asd_r
  
def f7(t,con,idd): # average section distance between prerequsite concepts
  asd_p = 0.0
  ctr = 0
  
  prerequsite=[]
  dir_in='./5_Link/'+con.replace("/","")
  f1 = open(dir_in+'_inlink.txt', 'r')
  for l1 in f1:
    prerequsite.append(l1.lower().strip())
  f1.close()
  
  gidd = idd.split("_")[0]+"_"+idd.split("_")[1]
  infile='./6_sFeature/'+t+'.json'
  with open(infile) as json_file:
    data = json.load(json_file)
    for d in data:
      if d['id'].startswith(gidd):
        for tt in d['topics']:
          if tt in prerequsite:
            asd_p += dist(d['id'],idd)
            ctr += 1
  if ctr >= 1:
    asd_p = asd_p / ctr
  return asd_p
  
def f8(t,con,idd): # average section distance between dependent concepts
  asd_d = 0.0
  ctr = 0
  
  dependent=[]
  dir_in='./5_Link/'+con.replace("/","")
  f1 = open(dir_in+'_outlink.txt', 'r')
  for l1 in f1:
    dependent.append(l1.lower().strip())
  f1.close()
  
  gidd = idd.split("_")[0]+"_"+idd.split("_")[1]
  infile='./6_sFeature/'+t+'.json'
  with open(infile) as json_file:
    data = json.load(json_file)
    for d in data:
      if d['id'].startswith(gidd):
        for tt in d['topics']:
          if tt in dependent:
            asd_d += dist(d['id'],idd)
            ctr += 1
  if ctr >= 1:
    asd_d = asd_d / ctr
  return asd_d
  
def f9(con): # number of relevant components absent in section
  rc = 0
  return rc

def run(t):
  dump_file={}
  dump_file['examples']=[]
  out_dir_name="7_cFeature/"
  if not os.path.exists(out_dir_name):
    os.makedirs(out_dir_name)
  infile='./6_sFeature/'+t+'.json'
  outfile='./7_cFeature/'+t+'.json'
  with open(infile) as json_file:
    data = json.load(json_file)
    for d in data:
      idd=d['id']
      con=d['topics']
      cf1 = []
      cf2 = []
      cf3 = []
      cf4 = []
      cf5 = []
      cf6 = []
      cf7 = []
      cf8 = []
      cf9 = []
      for c in con:
        cf1.append(round(f1(c,con),2))
        cf2.append(round(f2(c,con),2))
        cf3.append(round(f3(c,con),2))
        cf4.append(round(f4(c,con),2))
        cf5.append(round(f5(t,c,idd),2))
        cf6.append(round(f6(t,c,idd),2))
        cf7.append(round(f7(t,c,idd),2))
        cf8.append(round(f8(t,c,idd),2))
        cf9.append(round(f9(c),2))
      dump_file['examples'].append({"id":idd,"text":d['text'],"topics":d['topics'],"mentions":d['mentions'],"sf1":d['sf1'],"sf2":d['sf2'],"sf3":d['sf3'],"sf4":d['sf4'],"cf1":cf1,"cf2":cf2,"cf3":cf3,"cf4":cf4,"cf5":cf5,"cf6":cf6,"cf7":cf7,"cf8":cf8})
  with open(outfile, 'w') as ofile:
    json.dump(dump_file['examples'], ofile, indent=4)

def main(): #main for menu
  start_time = time.time()
  topics = ['Physics','Chemistry', 'Mathematics','Biology','Science','Geography','Economics']
  #topics = ['test']
  for t in topics:
    print(t)
    run(t)
  print("Exuction time: ", (time.time() - start_time))

if __name__ == '__main__':
  main()

