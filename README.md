# Chinese Hate Speech Feature Selection

This aims to explore potential features of hate speech in Chinese (simplified, traditional) and Cantonese that could be useful for future classification models. I scrapped relevant Tweets using a keyword approach as my dataset. A total of 10 features were extracted from the Tweets, and they fall into the following 4 categories: 

1. Semantic Features 
2. Sentiments 
3. Lexcal Features 
4. Linguistics Features 

For details of the features please see section [Feature Extraction](#feature-extraction)

I use Mutual Information and Logistic Regression to determine the relevance or the predictability or the features extracted. Results show that semantic familarity and profanity words are the most powerful features to predict hate speech. See [Feature Selection](#feature_selection) for details of the results.


Please downlaod the Chinese w2v corpus in here: https://ai.tencent.com/ailab/nlp/en/embedding.html
# Corpus 
# Feature Extraction
# Feature Selection
# Utility Files 
1. util.py
2. toy.tsv
A toy dataset (which means that the data inlcuded is manufactured) created for the purpose of demonstration. Please contact me for the actual dataset.
3. keyword.txt
Contains the keywords used to scrap Tweets. Please note it is by no means an exhaustive list. 
4. profane_words.txt
Contains profane words in Mandarin and Cantonese that are used as one of the features.
5. additional_words.txt
Combine the profane words and the keywords in one single list, and formated particularly for the use of the Chinese processing library Jieba.
6. NTUSD_negative_sentiment.txt, NTUSD_negative_sentiment.txt
contain positive and negative sentiment words adopted from [National Taiwan University Semantic Dictionary](https://rdrr.io/rforge/tmcn/man/NTUSD.html).
8. clustering.py
Reads a tsv file and uses a pretrained w2v model to create 7 files of clusters using K-means Clustering. An elbow graph is created as well to pre determine the suitable number of clusters. 
5.


# toy.tsv
modified tab seperated file with each line[0] as tweet, line[1] as the label

# clusters_sample
a folder that contains pre-produced cluster files 

# clustering.py 
reads a tsv file and uses a pretrained w2v model to create 7 files of cluster words. An elbow graph is created as well to pre determine the suitable number of clusters. 

Example usage:
`python3 clustering.py --tsv_file toy.tsv --w2v_file Tencent_AILab_ChineseEmbedding.txt`

# create_combined_cluster_file.py
reads the relevant cluster files <default to be cluster0,3,4 and 7> to combine all words from the files into one single file called combined_cluster.txt

Example usage:
If you have already run the clustering.py and created cluster files:
`python3 create_combined_cluster_file.py --cluster_files cluster0.txt cluster3.txt cluster4.txt cluster7.txt`
If you have not run the clustering.py, please use the pre-produced cluster files in the clusters_sample folder:
`python3 create_combined_cluster_file.py --cluster_files clusters_sample/cluster0.txt clusters_sample/cluster3.txt clusters_sample/cluster4.txt clusters_sample/cluster7.txt`

# feature_extraction.py
reads a tsv file and a combined clustered file to create a csv file called toyset.csv for feature testing 

Example usage: 
If you have run the cluster.py and create_combined_cluster_file.py (which creates the combined_cluster.txt)
`python3 feature_extraction.py --tsv_file toy.tsv --w2v_file Tencent_AILab_ChineseEmbedding.txt --clustered_word_file combined_cluster.txt` 

if you have not run clustering.py and create_combined_cluster.py, please used the pre-produced files in the clusters_sample folder
`Python3 feature_extraction.py --tsv_file toy.tsv --w2v_file Tencent_AILab_ChineseEmbedding.txt --clustered_word_file clusters_sample/combined_cluster.txt`

# feature_testing.py
reads a csv file created from feature_extraction.py to get the mutual information and logistic regression scores

Example usage:
`python3 feature_testing.py --csv_file toyset.csv`

# keywords.txt
a newline separated file for the hate group terms

# profane_words.txt
a newline separated file for the profane words

# additional_words.txt
a newline separated file for the additional words added to the Chinese processing library 

# NTUSD_negative_sentiment.txt, NTUSD_negative_sentiment.txt
newline separated files for sentiment words





