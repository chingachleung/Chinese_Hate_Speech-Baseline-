import argparse
from util import process_tweet,read_keywords,read_profanity,chinese_segmenter,read_sentiment, get_distributions
from gensim.models import KeyedVectors
import re
import synonyms
import jieba
import jieba.posseg as pseg
import csv

#global variables
keywords = read_keywords()
profane_words = read_profanity()

def determine_similarity(cluster, seg_tweet):
    """
    :param cluster: list of words
    :param seg_tweet: segmented tweet
    :return normalized number of similar words to the words in the cluster
    """
    # similarity score threshold = 0.5
    # todo : proved in k-means
    num = 0
    length = len(seg_tweet)
    #print(length)
    for seg in seg_tweet:
        for word in cluster:
            #0.9 as the threshold for similarity
            if synonyms.compare(seg,word) >= 0.9:
                num += 1
    num = num / length
    return num


def out_of_vocab(segmented_tweet, model):
    """
    calculates the number of oov words in a tweet, normalized by number of total words
    :param segmented_tweet: a list of words in a tweet
    """
    #todo: should be loaded in main
    #todo: proved
    sent_len = len(segmented_tweet)
    num = 0
    for word in segmented_tweet:
        if word not in model:
            print(word)
            num += 1
    norm_num = num/sent_len
    return norm_num

def num_of_punctuations(sentence):
    """
    :param tweet: unprocessed tweet
    :return punc_num_list : number of question marks/exclamations(normalied by sent length)
    """
    punctuations = ['?', '!','？','！']
    num = 0
    #todo: proved in kmeans
    #define sentence length by the number of characters
    chinese_chars=  re.findall(r'[\u4E00-\u9FFF]',sentence)
    sent_len = len(chinese_chars)
    for word in sentence:
        if word in punctuations:
            num += 1
    norm_num = num/sent_len
    return norm_num

def sentence_length(sentence):
    """
    :param tweet: unprocessed
    :return sent_len: number of chinese characters
    """
    #todo:proved
    #define sentence length by the number of characters
    chinese_chars=  re.findall(r'[\u4E00-\u9FFF]',sentence)
    sent_len = len(chinese_chars)
    return sent_len

def check_tone(sentence):
    """
    :param tweet: unprocessed
    :return norm_num: number of assertive,over-generalizing words , normalized
    """
    #todo:proved
    tone_word_list = ['全是','全部','凡是','都是','通通','统统','全都',
                      '就是','真是','真的','一定','必然','必定','肯定',
                      '不如','比','都不如','一样','最','果然是','就知道']
    num = 0
    sent_len = sentence_length(sentence)
    #print(f'sent len:{sent_len}')
    for word in tone_word_list:
        if word in sentence:
            print(word)
            num += 1
    #print(f'print num of tone words : {num}')
    norm_num = num/sent_len
    return norm_num


def sentence_final_particle(sentence):
    """
    sentence particles can change tone of a speaker

    :param unprocssed  tweet
    :return normalized  number of sentence particles in the sentence
    """
    #todo : proved
    particles = ['吧','呢','哦','啊','啦']
    sent_len = len(sentence)
    num = 0
    for c in sentence:
        if c in particles:
            num += 1

    return num /sent_len

def profanity_proximity(sentence):
    """
    :param sentence :unprocessed tweet
    :return Bool: whether there is a profane word nearby a nz (proper noun, which includes keywords)
    """
    words = pseg.lcut(sentence) # return a list of pair objects: (word, POS)
    print(f' words: {words}')
    #print(words)
    window=5

    #todo: proved
    for x, y in enumerate(words):
        print(f'the y of words: {y}')
        print(y)
        if 'nz' in y.flag: # check if the nz is put in the library yet
            print(f'the y.flag : {y.flag}')
            surroundings = [pair.word for pair in words[max(x - window, 0):x + window + 1]]
            print(surroundings)
            for w in surroundings:
                if w in profane_words:
                    print(w)
                    return True
    return False

def profanity_count(sentence):
    """
    :param unprocessed tweet
    :return normalized count of the number of profane words
    """
    chinese_chars = re.findall(r'[\u4E00-\u9FFF]', sentence)
    sent_len = len(chinese_chars)
    num = 0
    segmented_words =[word for word in chinese_segmenter(sentence)] # why complicate this
    print(segmented_words)
    for word in segmented_words:
        if word in profane_words:
            num+=1
    print(num)
    print(sent_len)
    return (num/sent_len)

def analyze_sentiment(sentence,p_s,ne_s):
    """
    :param sentence, p_s, ne_s: unprocessed sentence,  postive sentiment list, ne_s negative sentiment list
    :return num : higher number, the more positive
    """
    # probably vern slow since you will run on the two whole scripts...
    num = 0
    for word in p_s:
        if word in sentence:
            print(word)
            num +=1
    for word in ne_s:
        if word in sentence:
            print(word)
            num -=1
    return num

def othering_langauge(sentence):
    """
    :param sentence: unprocessed tweet
    :return normalized number of othering combination
    """

    out_group = ['你','你们','他','他们','它们','它','她们','她','牠','牠们'] #you, him, her, it(refers to animals), them
    sent_len = len(sentence)
    out_num = 0
    segmented = chinese_segmenter(sentence)
    for w in segmented:
        if w in out_group:
            out_num += 1
    return out_num/sent_len

def main(tsv_file,w2v_file,clustered_word_file):
    model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)

    #add additional words (profane and keywors) to the existing libraries
    jieba.load_userdict('additional_words.txt')
    
    #return a list of sentimental words
    positive_sentiments = read_sentiment('NTUSD_positive_sentiment.txt') 
    negative_sentiments = read_sentiment('NTUSD_negative_sentiment.txt')


    total_hs, total_non_hs, total_tweets , total_labels = get_distributions(tsv_file)

    #returns a list of segmented, processed tweets
    processed_tweets = []
    for tweet in total_tweets:
        processed_tweet = process_tweet(tweet)
        processed_tweets.append(processed_tweet)

    #returns a list of words from a cluster file
    clustered_words = []
    #todo should add that if you dont wish run the clusters - please changed the --clustered_word_file argument with
    #todo clusters_sample/combined_clusted.txt
    with open(clustered_word_file, 'r') as file:
        for line in file:
            word = line.strip()
            clustered_words.append(word)

    # feature lists
    similarity = []
    oov_values = []
    punc_nums = []
    sent_lengths = []
    tone_values = []
    particles = []
    profane_proximity =[]
    profanity_num = []
    sentiments = []
    othering = []


    for tweet in processed_tweets:
        similarity.append(determine_similarity(clustered_words,tweet))
        oov_values.append(out_of_vocab(tweet,model))


    for tweet in total_tweets:
        punc_nums.append(num_of_punctuations(tweet))
        sent_lengths.append(sentence_length(tweet))
        tone_values.append(check_tone(tweet))
        particles.append(sentence_final_particle(tweet))
        profane_proximity.append(profanity_proximity(tweet))
        profanity_num.append(profanity_count(tweet))
        sentiments.append(analyze_sentiment(tweet,positive_sentiments,negative_sentiments))
        othering.append(othering_langauge(tweet))

    #create a csv file of the features that will be used for testing
    with open('toyset.csv','w', newline="") as f:
        writer = csv.writer(f)
        writer.writerow(['class','similarity','oov','punc',
                         'sent_length','tone','particles','profanity proximity', 'profane num','sentiment','othering'])
        writer.writerows(zip(total_labels,similarity
                             ,oov_values,
                             punc_nums,sent_lengths,tone_values,
                             particles,profane_proximity,profanity_num,sentiments,othering))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature extraction')
    parser.add_argument("--tsv_file", type=str, default="toy.tsv",
                        help='tab seperated tweet data')
    parser.add_argument("--w2v_file", type=str, default="Tencent_AILab_ChineseEmbedding.txt",
                        help='chinese word2vecs embedding file')
    parser.add_argument("--clustered_word_file", type=str, default="combined_cluster.txt",
                        help='relevant words from different clusters combined')
    args = parser.parse_args()

    main(args.tsv_file,args.w2v_file,args.clustered_word_file)

#sample usage
#python3 feature_extraction.py --tsv_file toy.tsv --w2v_file Tencent_AILab_ChineseEmbedding.txt