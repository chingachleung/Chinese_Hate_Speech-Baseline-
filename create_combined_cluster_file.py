# combine all words in the relevant clusters into one file
#the relevant clusters are 0,3,4,7
import argparse
def combine_clusters(list_of_cluster_files):
    """
    :param list of cluster file: a list of relevant clusters
    :return combined_cluster.txt: a text file with all the words from the clusters

    """
    with open('combined_cluster.txt', 'w') as file:
        for cluster in list_of_cluster_files:
            with open(cluster, 'r', encoding='utf-8') as cf:
                for line in cf:
                    if line.strip():
                        file.write(line.strip())
                        file.write('\n')

def main(*arg):
    list_of_cluster_files = arg[0]
    combine_clusters(list_of_cluster_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='create different cluster files')
    parser.add_argument("--cluster_files", type=str,nargs="+", default=['cluster0.txt','cluster3.txt','cluster4.txt','cluster7.txt'],
                        help='a list of cluster files')
    args = parser.parse_args()

    main(args.cluster_files)

    #combine_clusters(list_of_cluster_files)
