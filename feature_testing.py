import pandas as pd
import numpy as np
import argparse

def calculate_MI(csv_file):
    """
    :param csv_file contains features and labels
    :return mutual information of features
    """
    df = pd.read_csv(csv_file)
    feature_data = df.drop(labels=['class'], axis=1)
    labels = df['class']
    from sklearn.feature_selection import mutual_info_classif
    mutual_info = mutual_info_classif(feature_data, labels)
    # better layout
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = feature_data.columns
    mutual_info.sort_values(ascending=False)
    return mutual_info


def log_regression(csv_file):
    """
    :param csv_file:
    : return precision, recall and f1 scores

    """

    df = pd.read_csv(csv_file)
    labels = df['class'].to_list()
    simi = df['similarity'].to_list()
    oov = df['oov'].to_list()
    punc = df['punc'].to_list()
    sent_len = df['sent_length'].to_list()
    tone = df['tone'].to_list()
    particles = df['particles'].to_list()
    profan_pro = df['profanity proximity'].to_list()
    profan_num = df['profane num'].to_list()
    sentiment = df['sentiment'].to_list()
    othering = df['othering'].to_list()
    features = [simi, oov, punc, sent_len, tone, particles, profan_pro, profan_num, sentiment, othering]
    observations = len(labels)
    X = np.zeros((observations, len(features)))
    for i, label in enumerate(labels):
        for j, metric in enumerate(features):
            X[i, j] = metric[i]
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, precision_score, recall_score
    model = LogisticRegression()
    model.fit(X, labels)
    pred = model.predict(X)
    p = precision_score(labels, pred, pos_label='HS')
    r = recall_score(labels, pred, pos_label='HS')
    f = f1_score(labels, pred, pos_label='HS')
    return p, r, f

def main(csv_file):
    mutual_information = calculate_MI(csv_file)
    p, r , f = log_regression(csv_file)
    print(f"mutual information of different features:\n{mutual_information}")
    print(f"precision score: {p}, recall score: {r}, f1 socre: {f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='feature extraction')
    parser.add_argument("--csv_file", type=str, default="toyset.csv",
                        help='csv with columns of features and label')
    args = parser.parse_args()

    main(args.csv_file)