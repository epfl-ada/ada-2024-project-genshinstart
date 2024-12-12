import os
import csv
import pandas as pd

def load_ai_paths(folder, ori_dest):
    ai_paths = {}
    for pair in ori_dest:
        start_article, target_article = pair
        file = f'{folder}/{start_article}_{target_article}.csv' 
        if os.path.exists(file):
            with open(file, 'r') as f:
                paths = list(csv.reader(f))
                paths = [path for path in paths if path != [] and path[-1]==target_article]
                ai_paths[pair] = paths
    return ai_paths

def load_human_paths(ori_dest):
    human_paths = {}
    paths_df = pd.read_csv("data/wikispeedia_paths-and-graph/paths_finished.tsv", sep="\t", comment='#', header=None,
                       names=["hashedIpAddress", "timestamp", "durationInSec", "path", "rating"])
    for pair in ori_dest:
        human_paths[pair] = []
    for index, row in paths_df.iterrows():
        path = row['path']
        path = path.split(';')
        start_article = path[0]
        target_article = path[-1]
        if (start_article, target_article) in ori_dest:
            human_paths[(start_article, target_article)].append(path)
    return human_paths