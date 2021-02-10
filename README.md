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
Transcript data for 27 textbooks from 7 subjects (Physics, Chemistry, Biology, Mathematics, Science, Geography and Economics) across 6-12 grades collected from https://ncert.nic.in. These data are collected and stored in 1_PDF folder in PDF format. You can download the data from https://drive.google.com/drive/folders/1bfzMKe-GATR-2rucQGPipN1qmFCCeH-S?usp=sharing
## 2. Preprocessing:
Textbook contents (PDFs) are converted into TXT format and pre-processed by removing spurious data (appendix, exercise etc.). The code '1_convert.py' converts and preprocesses the data from folder '1_PDF' and stores in '2_Text' folder in TXT format.
## 3. Segmentation:
The textbook contents are segmented into textbook sections. The code '2_section.py' segments transcripts from '2_Text' folder and stores in '3_Section' folder.
## 4. Concept Extraction:
A. Key-concepts are extracted for each textbook sections. The code '3_concept.py' extracts the topics and stores in '4_Concept' folder in JSON format. We create 'uConcepts.txt' to store all the unique concepts. We also create 7 different files (with name of the corrsponding grade-levels, e.g., '6.txt' for grade-level 6) to store the concepts, used in the textbooks asscoiated to specific grade-levels.

B. Wikipedia links are extracted for the extracted concepts. The code '4_wiki.py' extracts the inlinks and outlinks, and stores in '4_Link' folder in TXT format.

C. The code '5_aspect.py' combines the inlinks and outlinks for randomly selected 100 concepts, and stores in '5_Aspect' folder in TXT format. This folder contains three files: 'Related.txt,' 'Prerequisite.txt,' and 'Dependent.txt.' We assigned annotatotrs to tag these concepts and their aspcts as right/wrong, and these annotations are stored as 'GS_Related.txt,' 'GS_Prerequisite.txt,' and 'GS_Dependent.txt' (under folder 'GS' available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing).

## 5. Deficiency Diagnosis:
A. The concepts from '4_Concept' folder are shown to annotators and they are asked to tag the cooresponding deficiency. This annotation is stored as 'GS_Deficiency.txt' (under folder 'GS' available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing), made by combining 'GS1_Deficiency.txt' and 'GS1_Deficiency.txt.'

B. '6_sfeature.py' extracts the section-specfic features and stores in '6_sfeature' folder. Similarly, '7_cfeature.py' extracts the concept=specific features and stores in '7_cfeature' folder. Combining this features values with the annotation lables from 'GS_Deficiency.txt', we create a 'feature.txt' representing the feature vector and label.

C. Dataset from 'feature.txt' are divided in 3 parts: train, validate and test sets. '8_Deficiency.py' trains the deficiency identification model with train set and analyzes the performance of the features over the validation set to select the optimal set of features. Using those features, the deficient concepts are determined for the test set. The final performance of the deficiency identification model is also obtained. The predicted deficiency information is stored in '8_Deficiency' folder in TXT format.
## 6. Query Generation:
'9_Query.py' generates the queries for the concepts based on their context and deficiencies. These queries are stored in '9_Query' folder.
## 7. Textbook Augmentation:
For each query, '10_Retrieval.py' extracts the relvant augnmentations in the form of QA pairs and links them to the textbooks. These augmentations are stored in '10_Retrieval' folder in JSON format. These augmentations are shown to the annotators and asked to tag their relevance. This annotation is stored as 'GS_Augmentation.txt' (under folder 'GS' available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing), made by combining 'GS1_Augmentation.txt' and 'GS2_Augmentation.txt.'

## Prereqisites:
Here is a list of python libraries. Install them befoe running the codes:
* wikipedia-api
* whoosh
* tagme

## How to run:
Run 'main.py' which offers a menu-based control to execute each of the above-mentioned modules. If any of the required python libraries are not installed, associated errors will be generated. Install those libraries to run the codes successfully.

# Contacts
In case of any queries, you can reach us at kghosh.cs@iitkgp.ac.in
