# Chinese Hate Speech Feature Selection

This aims to explore potential features of hate speech in Chinese (simplified, traditional) and Cantonese that could be useful for future classification models. I scrapped relevant Tweets using a keyword approach as my dataset. A total of 10 features were extracted from the Tweets, and they fall into the following 4 categories: 

1. Semantic Features 
2. Sentiments 
3. Lexcal Features 
4. Linguistics Features 

For details of the features please see section [Feature Selection](#feature-selection)


Please downlaod the Chinese w2v corpus in here: https://ai.tencent.com/ailab/nlp/en/embedding.html
Please also feel free to use other smaller Chinese embedding corpora since loading this corpus takes quite a while.

# Feature Selection
# util.py 
contains 5 functions which will be used in other files

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





