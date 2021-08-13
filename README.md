# Augmentations for Textbooks

# Introduction:
This project presents a method for recommending augmentations against conceptual gaps in the textbooks. Question Answer (QA) pairs from community question-answering (cQA) forums are noted to offer a precise yet comprehensive illustration of concepts with compelling examples. Our proposed method retrieves QA pairs for a target concept to suggest two types of augmentations: basic and supplementary. Basic augmentations are suggested for the concepts for which a textbook lacks fundamental references.

The proposed textbook augmentation system has been realized using four major modules: (i) concept extraction, (ii) deficiency diagnosis, (iii) query generation and (iv) textbook augmentation.

1. Concept Extraction: Key-concepts are extracted from the textbook sections using an entity annotation service (TAGME).

2. Deficiency Diagnosis: The deficient concepts are diagnosed.

3. Query Generation: BVased on the deficiency, the concepts are merged with their context to form a keyword-based query.

4. Textbook Augmentation: Using the queries, a set of QA pairs are retrieved from the associated QA archives.

# Steps:

## 1. Data Collection:
A. Textbook data: 27 textbooks from 7 subjects (Physics, Chemistry, Biology, Mathematics, Science, Geography and Economics) across grade-levels 6-12 are collected from https://ncert.nic.in. These data are stored in 1_PDF folder in PDF format. You can download the data from https://drive.google.com/drive/folders/1bfzMKe-GATR-2rucQGPipN1qmFCCeH-S?usp=sharing

B. QA data: 6 different Stack Exchange sites (Mathematics, Physics, Chemistry, Biology, Earth Science, and Economics) are dumped as 7 XML files (Physics, Chemistry, Biology, Mathematics, Science, Geography and Economics) under 'QA' folder. You can download the data from https://drive.google.com/drive/folders/1bfzMKe-GATR-2rucQGPipN1qmFCCeH-S?usp=sharing
 
## 2. Preprocessing:
Textbook contents (PDFs) are converted into TXT format and pre-processed by removing spurious data (appendix, exercise etc.). The code '1_convert.py' converts and preprocesses the data from folder '1_PDF' and stores in '2_Text' folder in TXT format.

## 3. Segmentation:
The textbook contents are segmented into textbook sections. The code '2_section.py' segments the textbook contents from '2_Text' folder and stores in '3_Section' folder.

## 4. Concept Extraction:
Key-concepts are extracted for each textbook sections. The code '3_concept.py' extracts the concepts and stores in '4_Concept' folder in JSON format. We create 'uConcepts.txt' to store all the unique concepts for all the subjects. We also create 7 different files (with name of the corrsponding grade-levels, e.g., '6.txt' for grade-level 6) to store the concepts, used in the textbooks asscoiated to specific grade-levels.

## 4. Aspects:
A. Wikipedia links are extracted for the extracted concepts. The code '4_wiki.py' extracts the inlinks and outlinks, and stores in '5_Link' folder in TXT format.

B. The code '4_wiki.py' reads 'uConcepts.txt' and combines the inlinks and outlinks for the read concepts. This combination is written for randomly selected 100 concepts in 'GS' folder in TXT format as three files: 'Related.txt,' 'Prerequisite.txt,' and 'Dependent.txt.' We modify these files by assigning an annotator to tag these concepts and their aspcts as right/wrong. The annotated/modified files are available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing.

## 5. Deficiency Diagnosis:
A. The concepts from '4_Concept' folder are shown to annotators and they are asked to tag the cooresponding deficiency. This annotation is stored as 'GS_Deficiency.txt' (under folder 'GS' available at https://drive.google.com/drive/folders/1sbZoJzqcABbMce9uj1AojsQbDOqjCQaq?usp=sharing).

B. '5_sfeature.py' extracts the section-specfic features and stores in '6_sfeature' folder.

C. Similarly, '6_cfeature.py' extracts the concept=specific features and stores in '7_cfeature' folder.

D. Combining this features values with the annotation lables from 'GS_Deficiency.txt', code '7_Deficiency.py' creates two files 'feature1.csv' and 'feature2.csv'. 'feature1.csv' / 'feature2.csv' is the combination of featue vector and labels for baseline / proposed deficiency diagnosis module.

E. '7_Deficiency.py' trains, validates and tests the deficiency diagnosis module. It shows the final and subject-wise performance of the deficiency module in details.

## 6. Query Generation:
A. From each of the subjects and grade-levels, '8_Query.py' generates the queries and stores in '9_Query' in JSON format.

B. From each of the subjects, we randomly sample 500 queries from these sets of queries in '9_Query' in TXT format.

## 7. Textbook Augmentation:
A. For each query, '9_Retrieval.py' extracts augmentations (question id, title, body, tags and best accepted answer) from the QAs and stores under 'QA' folder 'QA.txt'.

B. '9_Retrieval.py' retrieves, re-ranks and filters these augmentations for the queries, generated in the previous step and the fectched augmentations are stored under '10_Retrieval' folder as 'RT.txt', 'RR.txt' and 'AUG.txt', respectively.

C. '9_Retrieval.py' assesses these retrieved augmentations against 'GS_Augmentation.txt' using six standard metrics: MAP (mean average precision), MRR (Mean reciprocal rank), RP (R-precision), P@1 (Precision at 1), P@5 (Precision at 5), and P@10 (Precision at 10). You can download the data from https://drive.google.com/drive/folders/1bfzMKe-GATR-2rucQGPipN1qmFCCeH-S?usp=sharing

## 8. Data Stats & Annotation Quality:
'10_get_stat.py' (a) offers statistics on the textbook data and QA data, (b) assesses the annotation quality for gold-standards for deficiency diagnosis and textbook augmentation.

## 9. Augmentations for Interface:
'11_augmentation.py' generates the augmentated textbooks where the concepts are linked with augmentations, directly.

## Prereqisites:
Here is a list of python libraries. Install them befoe running the codes:
* Wikipedia-api
* NLTK
* Spacy
* Numpy
* Sling
* Tagme
* 
* 
* 

# Contacts
In case of any queries, you can reach us at kghosh.cs@iitkgp.ac.in

# Cite
<!---
If this work is helpful for your research, please cite our paper 'Remediating Textbook Deficiencies by Leveraging Community Question Answers: A Machine Learning-based Approach' available at .

    @article{ghosh2022remediating,
        title = "Remediating Textbook Deficiencies by Leveraging Community Question Answers: A Machine Learning-based Approach",
        journal = "Education and Information Technologies",
        year = "2021",
        doi = "",
        author = "Krishnendu Ghosh, Plaban Kumar Bhowmick and Pawan Goyal ",
        keywords = "Concept extraction, Deficiency diagnosis, Query generation, Question retrieval, Textbook augmentation"
    }
---!>
A similar work on augmenting video lectures is discussed in our paper 'Augmenting Video Lectures: Identifying Off-topic Concepts and Linking to Relevant Video Lecture Segments' available at https://link.springer.com/article/10.1007/s40593-021-00257-z.

    @article{ghosh2021augmenting,
        title = "Augmenting Video Lectures: Identifying Off-topic Concepts and Linking to Relevant Video Lecture Segments",
        journal = "International Journal of Artificial Intelligence in Education",
        year = "2021",
        doi = "https://doi.org/10.1007/s40593-021-00257-z",
        url = "https://link.springer.com/article/10.1007/s40593-021-00257-z",
        author = "Krishnendu Ghosh, Sharmila Reddy Nangi, Yashasvi Kanchugantla, Pavan Gopal Rayapati, Plaban Kumar Bhowmick and Pawan Goyal ",
        keywords = "Video lecture augmentation, Off-topic concept identification, MOOCs, Concept similarity, Community detection, Retrieval and re-ranking"
    }

The module on retrieving questions is discussed in details in our paper 'Using Re-Ranking to Boost Deep Learning Based Community Question Retrieval' available at https://dl.acm.org/doi/pdf/10.1145/3106426.3106442.

    @inproceedings{ghosh2017using,
    author = {Ghosh, Krishnendu and Bhowmick, Plaban Kumar and Goyal, Pawan},
    title = {Using Re-Ranking to Boost Deep Learning Based Community Question Retrieval},
    year = {2017},
    isbn = {9781450349512},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3106426.3106442},
    doi = {10.1145/3106426.3106442},
    booktitle = {Proceedings of the International Conference on Web Intelligence},
    pages = {807–814},
    numpages = {8},
    keywords = {question retrieval, re-ranking, community question answering},
    location = {Leipzig, Germany},
    series = {WI '17}
    }
