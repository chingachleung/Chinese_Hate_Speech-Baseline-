from util import process_tweet,read_keywords
from collections import Counter
from itertools import chain
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from util import get_distributions
import argparse

def get_frequent_words(hs_list, non_hs_list):

    """read two lists: a HS list with hateful tweets, and a non-H.S list with non hateful tweets
    return a list of words from the hs tweets that have higher normalized frequency than in non-hs
     threshold: minimum 5 occurences
    """

    key_words = read_keywords()
    segmented_processed_hs = []
    segmented_processed_non_hs = []

    #remove hate group keywords

    for tweet in hs_list:
        segmented_tweet = process_tweet(tweet)
        for word in segmented_tweet:
            if word in key_words:
                segmented_tweet.remove(word)
        segmented_processed_hs.append(segmented_tweet)
    print(f'length of processed hs : {len(segmented_processed_hs)}')

    for tweet in non_hs_list:
        segmented_tweet = process_tweet(tweet)
        for word in segmented_tweet:
            if word in key_words:
                segmented_tweet.remove(word)
        segmented_processed_non_hs.append(segmented_tweet)
    print(f'length of processed non hs : {len(segmented_processed_non_hs)}')

    hs_counter = Counter(chain.from_iterable(segmented_processed_hs))
    non_hs_counter = Counter(chain.from_iterable(segmented_processed_non_hs))
    print(f'printing the hs counter lengths: {len(hs_counter.keys())}')

    #only keep words that occur at least 5 times for the real dataset, but it breaks the 200 data point sample;
    #so value is set at 1 for demonstration purpose
    threshold_hs_counter = Counter({k:v for k,v in hs_counter.items() if v >= 1})
    threshold_non_hs_counter = Counter({k:v for k,v in non_hs_counter.items() if v >= 1})
    print(f'printing the hs counter threshold: {len(threshold_hs_counter.keys())}')

    hs_scale = 1
    non_hs_scale = (len(non_hs_list) / len(hs_list))

    #get normalized count of the hs words
    count_array1 = np.array([v for v in threshold_hs_counter.values()])
    norm_count_array1 = count_array1 / hs_scale
    word_list1 = [w for w in threshold_hs_counter.keys()]

    #words from non-hs that are above threshold
    word_list2 = [w for w in threshold_non_hs_counter.keys()]

    # pick words from the H.S keys only if it has higher normalized counts
    frequent_word_list = []
    for i, w in enumerate(word_list1):
        hs_norm_count = norm_count_array1[i]
        if w in word_list2:
            if hs_norm_count > (threshold_non_hs_counter[w]/non_hs_scale):
                frequent_word_list.append(w)
        else:
            frequent_word_list.append(w)
    return frequent_word_list


def create_word_clusters(frequent_word_list,model):
    """
    :param word counter:  word:frequency dict with keys from H.S that have higher frequency than its equvalent in non H.S
    :return clusters: default to be 8 for now
    """

    words = frequent_word_list
    word_vecs = []
    out_of_vocab = []

    for word in words:
        if word in model:
            word_vecs.append(model[word])
        else:
            out_of_vocab.append(word)
    X = np.array(word_vecs)

    #make sure label and word dimensions are the same
    for word in out_of_vocab:
        if word in words:
            words.remove(word)

    #elbow graph to determine the number of clusters
    wcss = []
    for i in range (1,9):
        kmeans = KMeans(n_clusters=i,init='k-means++',random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1,9),wcss)
    plt.title("the elbow method")
    plt.xlabel("number of clusters")
    plt.ylabel("WCSS")
    plt.show()

    # the graph shows  8 is good cluster numbers

    n_cluster = 8
    clf = KMeans(n_clusters=n_cluster,max_iter=10,init="k-means++",n_init=1).fit(X)
    labels = clf.predict(X) # predict which cluster a word should belong to [1,2,3,1,2,1,...]

    print(labels)
    print('printing the clusters now')
    cluster0 = []
    cluster1 = []
    cluster2 = []
    cluster3 = []
    cluster4 = []
    cluster5 = []
    cluster6= []
    cluster7 = []
    for i, word in enumerate(words):
        if labels[i] == 0:
            cluster0.append(word)
        elif labels[i] == 1:
            cluster1.append(word)
        elif labels[i] == 2:
            cluster2.append(word)
        elif labels[i] == 3:
            cluster3.append(word)
        elif labels[i] == 4:
            cluster4.append(word)
        elif labels[i] == 5:
            cluster5.append(word)
        elif labels[i] == 6:
            cluster6.append(word)
        elif labels[i] == 7:
            cluster7.append(word)

    #print the words in different clusters into different files
    with open('cluster0.txt','w') as f:
        for w in cluster0:
            f.write(w)
            f.write('\n')
    with open('cluster1.txt','w') as f:
        for w in cluster1:
            f.write(w)
            f.write('\n')
    with open('cluster2.txt','w') as f:
        for w in cluster2:
            f.write(w)
            f.write('\n')
    with open('cluster3.txt','w') as f:
        for w in cluster3:
            f.write(w)
            f.write('\n')
    with open('cluster4.txt','w') as f:
        for w in cluster4:
            f.write(w)
            f.write('\n')
    with open('cluster5.txt','w') as f:
        for w in cluster5:
            f.write(w)
            f.write('\n')
    with open('cluster6.txt','w') as f:
        for w in cluster6:
            f.write(w)
            f.write('\n')
    with open('cluster7.txt','w') as f:
        for w in cluster7:
            f.write(w)
            f.write('\n')

    # from first obsevations, only cluster0,cluster3, cluster4,cluster7 are useful;


def main(tsv_file, w2v_file):
    model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    hs, non_hs, total, labels = get_distributions(tsv_file)
    #find the more relevant word from H.S
    frequent_word_list = get_frequent_words(hs, non_hs)
    #create files of different clusters
    create_word_clusters(frequent_word_list, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create different cluster files')
    parser.add_argument("--tsv_file", type=str, default="toy.tsv",
                        help='tab seperated tweet data')
    parser.add_argument("--w2v_file", type=str, default="Tencent_AILab_ChineseEmbedding.txt",
                        help='chinese word2vecs embedding file')
    args = parser.parse_args()

    main(args.tsv_file,args.w2v_file)

