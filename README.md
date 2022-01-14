# Chinese Hate Speech Classification Task  (Baseline Model)

This task aims to explore potential features of hate speech in Chinese that could be useful for future more advanced classification tasks. The training dataset contains relevant Tweets I scrapped using a keyword approach. A total of 10 features were extracted from the Tweets, and they fall into the following 4 categories: 

1. Semantic Features 
2. Sentiments 
3. Lexical Features 
4. Linguistic Features 

For details of the features, please see section [Feature Extraction](#feature-extraction).

I used Mutual Information to determine which features are associated to hate speech, and used Logistic Regression to determine the predictability of the features extracted. See [Feature Selection](#feature-selection) for the results.


# Data 
I started with devlopeing a [hate speech lexcion](https://github.com/chingachleung/Chinese_Hate_seepch/blob/main/keywords.txt) based on multiple resources ([Hatebase.org](https://hatebase.org/), [Wiikpedia](https://zh.wikipedia.org/wiki/%E6%AD%A7%E8%A7%86%E8%AF%AD)) and retrieved Tweets using Twitter API that contain terms from the Lexicon. The Tweets in the dataset are in Simplified Chinese, Traditional Chinese or Cantonese. For this baseline model, I collected over 6000 Tweets. Around 25% percent of the Tweets are classifified as hate-speech, and 75% classified as non hate-speech. To inqure about the data, please contact me at wl607@georgetown.edu

# Feature Extraction

The following features are extracted from each Tweet:
1. Semantic similarity\
To compare semantic similarity, I used the [Tencent Chinese W2V Corpus](https://ai.tencent.com/ailab/nlp/en/embedding.html) for embeddings.
2. OOV counts
3. Punctuations 
4. Sentence lengths
5. Sentence final particles
6. Profanity count
7. Sentiment 
8. Othering language
9. Assertiveness
10. Profanity proximity
 
feature_extraction.py reads a tsv file and a combined clustered file to create a csv file called toyset.csv for feature testing.

Example usage: 
(You need to run the cluster.py and create_combined_cluster_file.py as part of pre-processing first.)
`python3 feature_extraction.py --tsv_file toy.tsv --w2v_file Tencent_AILab_ChineseEmbedding.txt --clustered_word_file combined_cluster.txt` 

# Feature Selection
The similarity and the profanity features score the highest in Mutual Information. The logistic regression trained on all the features yields a precision score of 0.569, recall score of 0.456 and F1 score of 0.506.

feature_testing.py reads a csv file created from feature_extraction.py to get the mutual information and logistic regression scores.\
Example usage:
`python3 feature_testing.py --csv_file toyset.csv`

# Utility Files 
1. util.py\
Uses Jieba and spaCy to pre-process Tweets.
3. toy.tsv\
A toy dataset created for the purpose of demonstration. They are NOT real data.
3. keyword.txt\
Contains the keywords used to scrap Tweets. Please note it is by no means an exhaustive list. 
4. profane_words.txt\
Contains profane words in Mandarin and Cantonese that are used as one of the features.
5. additional_words.txt\
Combines the profane words and the keywords in one single list, and formated particularly for the use of the Chinese processing library [Jieba](https://pypi.org/project/jieba/).
6. NTUSD_negative_sentiment.txt, NTUSD_negative_sentiment.txt\
Contain positive and negative sentiment words adopted from [National Taiwan University Semantic Dictionary](https://rdrr.io/rforge/tmcn/man/NTUSD.html).
7. clustering.py\
Reads a tsv file and uses a pretrained w2v model to create 7 files of clusters using K-means Clustering. An elbow graph is created as well to pre determine the suitable number of clusters.\
Example usage:
`python3 clustering.py --tsv_file toy.tsv --w2v_file Tencent_AILab_ChineseEmbedding.txt`
8. create_combined_cluster_file.py\
Reads the relevant cluster files <default to be cluster0,3,4 and 7> to combine all words from the files into one single file called combined_cluster.txt\
Example usage:
`python3 create_combined_cluster_file.py --cluster_files cluster0.txt cluster3.txt cluster4.txt cluster7.txt`

# Exciting News! 

I have expanded the dataset into over 10000 Tweets. Currently in preparation for a paper to be submitted to confereneces. Stay tuned for updates!




