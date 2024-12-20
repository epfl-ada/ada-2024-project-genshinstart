import pandas as pd
import networkx as nx
import csv
from tqdm import tqdm
import openai
import os
import urllib.parse

def play_wikispeedia_game(start_article, target_article, graph, client, model="gpt-4o-mini"):
    '''
    Generate a path from start_article to target_article using the model
    Usage: 
    play_wikispeedia_game('A', 'B', G, client)

    Parameters:
    start_article (str): The starting article
    target_article (str): The target article
    graph (networkx.DiGraph): The graph of the articles
    client (openai.Client): The OpenAI client
    model (str): The model to use, default is 'gpt-4o-mini'

    Returns:
    list: The path from start_article to target_article
    '''
    current_article = start_article
    path = [current_article]
    used_tokens = 0

    # start the conversation
    conversation = [
        {"role": "system", "content": "You are an expert in navigating articles. Your goal is to find the path as short as possible from the start article to the target article by selecting the provided links. You can also return to the previous article by replying 'back'. Please reply only with the name of the article you want to move to, with no explanation or marks."}, 
        {"role": "user", "content": f"Start at the article '{start_article}' and navigate to '{target_article}'."}
    ]
    
    length = 0
    previous_article_stack = [] # stack to store the previous article, to deal with 'back' command
    while current_article != target_article and length < 30:
        length += 1
        neighbors = list(graph.successors(current_article))
        if not neighbors:
            break
        
        # put the current state into the conversation
        conversation.append(
            {"role": "user", "content": f"You are currently at the article '{current_article}' and the target is '{target_article}'. The available links are: {', '.join(neighbors)}. Which article do you want to click on next?"}
        )

        # get the next article from the model
        response = client.chat.completions.create(
            model=model, 
            messages=conversation
        )
        used_tokens += response.usage.total_tokens
        next_article = response.choices[0].message.content

        # URL-encode the next article
        next_article = urllib.parse.unquote(next_article)
        next_article = urllib.parse.quote(next_article)

        retry = 0
        while next_article not in neighbors and next_article != 'back':
            # print('retry in article:', next_article)
            # print('neighbors:', neighbors)
            retry += 1
            conversation.append(
                {"role": "assistant", "content": f"Sorry, '{next_article}' is not a valid link in current article '{current_article}'. Please choose from the following links: {', '.join(neighbors)}."}
            )
            response = client.chat.completions.create(
                model=model, 
                messages=conversation
            )
            used_tokens += response.usage.total_tokens
            next_article = response.choices[0].message.content
            if retry > 3:
                break
        
        if retry > 3:
            return ['Multiple retries']
        
        # print(next_article)
        path.append(next_article)
        conversation.append({"role": "assistant", "content": next_article})

        if next_article == 'back':
            if previous_article_stack == []:
                return ['Back in the beginning']
            next_article = previous_article_stack.pop()
        else:
            previous_article_stack.append(current_article)
            current_article = next_article
    
    # print(f'Used tokens: {used_tokens}')
    if path[-1] != target_article:
        return ['Failed']
    return path

if __name__ == '__main__':
    # load the links file
    df = pd.read_csv('wikispeedia_paths-and-graph/links.tsv', sep='\t', comment='#', header=None, names=['origin', 'destination'])

    # Create a directed graph
    G = nx.from_pandas_edgelist(df, source='origin', target='destination', create_using=nx.DiGraph())

    # load the origin and destination pairs, in the first column
    ori_dest = pd.read_csv('ori_dest.csv').iloc[:, 0].tolist()
    ori_dest = [eval(pair) for pair in ori_dest]

    # Create OpenAI client
    api_key = "Your API key"
    client = openai.OpenAI(api_key = api_key)

    # play the game for each pair several times, and save the paths for each pair
    for pair in tqdm(ori_dest):
        start_article, target_article = pair
        paths = []
        for _ in range(5):
            path = play_wikispeedia_game(start_article, target_article, G, client)
            if path[-1] == target_article:
                paths.append(path)
            else:
                paths.append([path[-1]])
        
        if not os.path.exists('paths'):
            os.makedirs('paths')
        
        with open(f'paths/{start_article}_{target_article}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(paths)

        # print average path length
        err_count = paths.count(['Failed']) + paths.count(['Multiple retries']) + paths.count(['Back in the beginning'])
        if err_count == 5:
            print(f'No path found for {start_article} to {target_article}')
            continue
        avg_length = sum([len(path) for path in paths if path[-1] == target_article]) / (len(paths) - err_count)
        print(f'Average path length for {start_article} to {target_article}:', avg_length)
        print(f'Multiple retries: {paths.count(["Multiple retries"])}, Back in the beginning: {paths.count(["Back in the beginning"])}, Failed: {paths.count(["Failed"])}')          
            