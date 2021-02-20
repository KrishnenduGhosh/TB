# Augmentations for Textbooks

# Introduction:
This project presents a method for recommending augmentations against conceptual gaps in the textbooks. Question Answer (QA) pairs from community question-answering (cQA) forums are noted to offer a precise yet comprehensive illustration of concepts with compelling examples. Our proposed method retrieves QA pairs for a target concept to suggest two types of augmentations: basic and supplementary. Basic augmentations are suggested for the concepts for which a textbook lacks fundamental references.

The proposed textbook augmentation system has been realized using four major modules: (i) concept extraction, (ii) deficiency diagnosis, (iii) query generation and (iv) textbook augmentation.

1. Concept Extraction: Key-concepts are extracted from the textbook sections using an entity annotation service (TAGME).

2. Deficiency Diagnosis: The deficient concepts are diagnosed.

3. Query Generation: BVased on the deficiency, the concepts are merged with their context to form a keyword-based query.

4. Textbook Augmentation: Using the queries, a set of QA pairs are retrieved from the associated QA archives.

# Steps:

## 1. Data collection:
27 textbooks from 7 subjects (Physics, Chemistry, Biology, Mathematics, Science, Geography and Economics) across grade-levels 6-12 are collected from https://ncert.nic.in. These data are stored in 1_PDF folder in PDF format. You can download the data from https://drive.google.com/drive/folders/1bfzMKe-GATR-2rucQGPipN1qmFCCeH-S?usp=sharing

6 different Stack Exchange sites are dumped: Mathematics, Physics, Chemistry, Biology, Earth Science, and Economics and stored as XML files under 'QA' folder. '10_Retrieval.py' organizes these dumps and stores them as TXT files under 'QA' folder to:
* to remove QA pairs which misses any of: Question id, Question Title, Question Body, Answer, and Tags.
* to connect Questions with their best Accepted Answers present in the dump

## 2. Preprocessing:
Textbook contents (PDFs) are converted into TXT format and pre-processed by removing spurious data (appendix, exercise etc.). The code '1_convert.py' converts and preprocesses the data from folder '1_PDF' and stores in '2_Text' folder in TXT format.

## 3. Segmentation:
The textbook contents are segmented into textbook sections. The code '2_section.py' segments the textbook contents from '2_Text' folder and stores in '3_Section' folder.

## 4. Concept Extraction:
A. Key-concepts are extracted for each textbook sections. The code '3_concept.py' extracts the concepts and stores in '4_Concept' folder in JSON format. We create 'uConcepts.txt' to store all the unique concepts for all the subjects. We also create 7 different files (with name of the corrsponding grade-levels, e.g., '6.txt' for grade-level 6) to store the concepts, used in the textbooks asscoiated to specific grade-levels.

B. Wikipedia links are extracted for the extracted concepts. The code '4_wiki.py' extracts the inlinks and outlinks, and stores in '4_Link' folder in TXT format.

C. The code '5_aspect.py' combines the inlinks and outlinks for randomly selected 100 concepts, and stores in '5_Aspect' folder in TXT format. This folder contains three files: 'Related.txt,' 'Prerequisite.txt,' and 'Dependent.txt.' We assigned an annotator to tag these concepts and their aspcts as right/wrong, and these annotations are stored as 'GS_Related.txt,' 'GS_Prerequisite.txt,' and 'GS_Dependent.txt' (under folder 'GS' available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing).

## 5. Deficiency Diagnosis:
A. The concepts from '4_Concept' folder are shown to annotators and they are asked to tag the cooresponding deficiency. This annotation is stored as 'GS_Deficiency.txt' (under folder 'GS' available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing).

B. '6_sfeature.py' extracts the section-specfic features and stores in '6_sfeature' folder. Similarly, '7_cfeature.py' extracts the concept=specific features and stores in '7_cfeature' folder. Combining this features values with the annotation lables from 'GS_Deficiency.txt', we create a 'feature_deficiency.txt' representing the feature vector and label.

C. Dataset from 'feature_deficiency.txt' are divided in 3 parts: train, validate and test sets. '8_Deficiency.py' trains the deficiency identification model with train set and analyzes the performance of the features over the validation set to select the optimal set of features. Using those features, the deficient concepts are determined for the test set. The final performance of the deficiency identification model is also obtained.

## 6. Query Generation:
From each of the subjects and grade-levels, we randomly selected 2 concepts. This offers a set of 50 concepts. '9_Query.py' generates the queries for these 50 concepts based on their context and deficiencies. These queries are stored in '9_Query' folder as <Subject>_q.txt and <Grade>_q.txt.

## 7. Textbook Augmentation:
For each query, '10_Retrieval.py' extracts the relvant augnmentations in the form of QA pairs and links them to the textbooks. These augmentations are stored in '10_Retrieval' folder in JSON format. These augmentations are shown to the annotators and asked to tag their relevance. This annotation is stored as 'GS_Augmentation.txt' (under folder 'GS' available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing).

## Prereqisites:
Here is a list of python libraries. Install them befoe running the codes:
* Wikipedia-api
* NLTK
* Spacy
* Numpy
* 
* Tagme

## How to run:
Run 'main.py' which offers a menu-based control to execute each of the above-mentioned modules. If any of the required python libraries are not installed, associated errors will be generated. Install those libraries to run the codes successfully.

# Contacts
In case of any queries, you can reach us at kghosh.cs@iitkgp.ac.in
