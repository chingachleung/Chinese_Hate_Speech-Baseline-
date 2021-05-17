import jieba
from spacy.lang.zh.stop_words import STOP_WORDS
import re
import chinese_converter
#add additional stop words at the same time
STOP_WORDS |= {'「', '嘅', '啲', '系', '里','咁','唔','中','有没有','咩','佢',"妳们",'哋','乜'}

def get_distributions(tsv_file):
    """
    Reads a tab-separated tweet file and returns
    total_hs: a list of str(hateful tweet)
    total_non_hs: a list of str(non hateful tweet)
    total_tweet: a list of all the tweets
    total_labels: a list of all labels
    """

    total_hs = []
    total_non_hs = []
    total_tweets = []
    total_labels = []
    with open(tsv_file,'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            fields = line.split('\t')
            #make sure all tweets are simplified
            tweet = chinese_converter.to_simplified(fields[0])
            label = fields[1].strip()
            total_tweets.append(tweet)
            total_labels.append(label)
            if label == "HS":
                total_hs.append(tweet)
            else:
                total_non_hs.append(tweet)

    print(f'number of hs, nhs, and total: {len(total_hs)}, {len(total_non_hs)},{len(total_tweets),len(total_labels)}')
    return (total_hs, total_non_hs, total_tweets,total_labels)

def chinese_segmenter(sent):
    """reads a sentence and uses jieba library to segment the sentence, and returns
    default segmented tweet: a list of tokens
    """
    # lcut returns a list, while cut returns a generator
    def_seg_list = jieba.lcut(sent,cut_all=False)
    return def_seg_list

def read_keywords(keyword_file='keywords.txt'):
    """load a newline separated text file of keywords used to collect tweets.
    Return a list.
    """
    keywords = []
    with open(keyword_file,'r') as f:
        for line in f:
            if line.strip():
                keywords.append(line.strip())
    return keywords

def read_profanity(keyword_file='profane_words.txt'):
    """load a newline separated text file of profane words.
    Return a list.
    """
    p_words = []
    with open(keyword_file,'r') as f:
        for line in f:
            if line.strip():
                p_words.append(line.strip())
    return p_words

def process_tweet(tweet):
    """
    :param tweet: unprocessed sentence
    segmented tweets: segmented tweet without punctuations,stop words, English words replaced with space
    """

    #applies to simplified chinese
    #todo cancel out this since it is in the main
    #keywords = read_keywords()

    #replace non-chinese words, digits,punctuations with space
    tweet = re.sub(r"[A-Za-z]|\d|\W"," ",tweet)
    segmented_tweet = chinese_segmenter(tweet)
    processed_tweet = []
    for token in segmented_tweet:
        if (token not in STOP_WORDS and token != " "):
            processed_tweet.append(token)
    return processed_tweet

def read_sentiment(text_file):
    """load a newline separated text file of sentiment words.
    Return a list of negative/positive words
    """
    #todo need to be checked
    sentiment_words  = []
    with open(text_file,'r',encoding='utf8') as f:
        for line in f:
            if line.strip():
                sentiment_words.append(line.strip())
    return sentiment_words
