# Augmentations for Textbooks

# Introduction:
This project presents a method for recommending augmentations against conceptual gaps in the textbooks. Question Answer (QA) pairs from community question-answering (cQA) forums are noted to offer a precise yet comprehensive illustration of concepts with compelling examples. Our proposed method retrieves QA pairs for a target concept to suggest two types of augmentations: basic and supplementary. Basic augmentations are suggested for the concepts for which a textbook lacks fundamental references.

The proposed textbook augmentation system has been realized using four major modules: (i) concept extraction, (ii) deficiency diagnosis, (iii) query generation and (iv) textbook augmentation.

1. Concept Extraction: The segmented video lectures are indexed using its concepts, extracted with an entity annotation service.

2. Deficiency Diagnosis: The task of predicting off-topic concepts is implemented as a community structure analysis problem on concept similarity graphs.

3. Query Generation: For each off-topic concept, appropriate video segments are fetched.

4. Textbook Augmentation: These initially retrieved video segments are further re-ranked so that the top-ranked video segment offers the most basic understanding of the target off-topic concept.

# Steps:
## 1. Data collection:
Transcript data for 27 textbooks from 7 subjects (Physics, Chemistry, Biology, Mathematics, Science, Geography and Economics) across 6-12 grades collected from https://ncert.nic.in. These data are collected and stored in 1_PDF folder in PDF format. You can download the data from https://drive.google.com/open?id=1KTWhbPk-N8_rz-p-wSIMo9nKKYWU7PU9
## 2. Preprocessing:
Textbook contents (PDFs) are converted into TXT format and pre-processed by removing spurious data (appendix, exercise etc.). The code '1_convert.py' converts and preprocesses the data from folder '1_PDF' and stores in '2_Text' folder.
## 3. Segmentation:
The tTextbook contents are segmented into textbook sections. The code '2_section.py' segments transcripts from '2_Text' folder and stores in '3_Section' folder.
## 4. Concept Extraction:
A. Key-concepts are extracted for each textbook sections. The code '3_concept.py' extracts the topics and stores in '4_Concept' folder in JSON format.

B. Wikipedia links are extracted for the extracted concepts. The code '4_Wiki.py' extracts the inlinks and outlinks, and stores in '4_Links' folder in JSON format.

C. Based on these links, aspects (prerequisite, dependent or related concepts) for a concept is determined. The code '5_Aspect.py' determines these aspects for all the concepts present in '4_Links' folder and stores the aspects in '4_Aspects' folder in JSON format. It also stores a small dataset of 100 concepts and their
aspects in 'Aspects.txt' for manual annotation. The annotator tagged these concepts and their aspcts as right/wrong, and this annotation is stored as 'GS_Aspects.txt' (available at https://drive.google.com/open?id=1peCDKd2u1xUuez5waN-2OgFRaSvUelh3).
## 5. Deficiency Diagnosis:
The textbook concepts are diagnosed for deficiencies. The code '4_off_predict.py' identifies the off-topics, stores them in '5_off' folder and also evaluates the concerned modules.
## 6. Query Generation:
Video lecture segments relevant to each of the off-topics are retrieved. The code '6_retrieval.py' retrieves the segments and stores in '6_Retrieved' folder in JSON format and as 'RT.txt' in '8_Retrieved/trec-eval/test' folder in TREC suggested text format. The '8_Result' folder is downloadable from https://drive.google.com/open?id=17-IxebyTtNsSXY98FfkTJWHK9goHhkOT which contains the folder 'trec-eval', providing the performance evaluation codes.
## 7. Textbook Augmentation:
A. Code '7_feature.py' extracts the features and stores them in 'rerank.txt' file under '7_Reranked' folder.

B. The extracted features are combined with the the labels (relevant or not) from 'GS.txt' by code '7_L2R.py' and stores in 'L2R.txt'. Running linear regression models, the code '7_L2R.py' further detemines the weights for the features.

C. The retrieved video lecture segments are reranked using code '7_rerank.py' where the learned weights are used. The reranked segments are stored in '7_Reranked' folder in JSON format and as 'RR.txt' in '8_Retrieved/trec-eval/test' folder in TREC suggested text format.
## 8. Evaluation:
A. The retrieved and reranked segmnets are shown to the annotators and their relevance are tagged. The gold standard is present in 'GS.txt' file. The file can be downloaded from https://drive.google.com/open?id=1sKfmBveCkUtaL_5cJqKG0li_z-c0wns4 .

B. The code '8_eval.py' evaluates the retrieval and re-ranking performance. The '8_Result' folder is downloadable from https://drive.google.com/open?id=17-IxebyTtNsSXY98FfkTJWHK9goHhkOT which contains the folder 'trec-eval', providing the performance evaluation codes.

# Run:
## Prepare the pre-requisites:
A. To run the above-mentioned codes, one needs a list of supporting files, as offered inside a 'lib' folder in the current directory. The 'lib' folder can be downloaded from https://drive.google.com/open?id=11PJ0Y-3RavS2F0B8lj247M5pK19fK11I.

B. Geckodriver is also required. Download this from https://drive.google.com/open?id=1Mf92NT_MNV-z2ZXVkkuneIGw7hLoe8n1 and place in the current directory. Export it in PATH before running the codes.
## Execute:
Finally, run 'main.py' which offers a menu-based control to execute each of the above-mentioned modules.

# Contacts
In case of any queries, you can reach us at kghosh.cs@iitkgp.ac.in
