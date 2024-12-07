# Generate paths from the origin and destination pairs that appear more than 10 times, using gpt
'''
The game is that links are given to the player, and the player has to navigate from the origin to the destination by clicking on the links. Please reply only with the name of the article you want to move to, with no explanation or marks.
'''

import pandas as pd
import networkx as nx
import csv
from tqdm import tqdm
import openai

def play_wikispeedia_game(start_article, target_article, graph, client, model="gpt-4o-mini"):
    current_article = start_article
    path = [current_article]
    used_tokens = 0
    conversation = [
        {"role": "system", "content": "You are playing a game where you navigate from one article to another by selecting the providing links. You can also return to the previous article by replying \"back\". The goal is to reach the target article in the fewest steps possible. Please reply only with the name of the article you want to move to, with no explanation or marks."},
        {"role": "user", "content": f"Start at the article '{start_article}' and navigate to '{target_article}'."}
    ]
    
    length = 0
    previous_article_stack = []
    while current_article != target_article and length < 15:
        length += 1
        neighbors = list(graph.successors(current_article))
        if not neighbors:
            break
        
        conversation.append(
            {"role": "user", "content": f"You are currently at the article '{current_article}'. The available links are: {', '.join(neighbors)}. Which article do you want to click on next?"}
        )

        response = client.chat.completions.create(
            model=model, 
            messages=conversation
        )
        used_tokens += response.usage.total_tokens
        next_article = response.choices[0].message.content

        retry = 0
        while next_article not in neighbors and next_article != 'back':
            print('retry in article:', next_article)
            print('neighbors:', neighbors)
            retry += 1
            conversation.append(
                {"role": "assistant", "content": f"Sorry, '{next_article}' is not a valid. Please choose from the following links: {', '.join(neighbors)}."}
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
            break
        
        print(next_article)
        path.append(next_article)
        conversation.append({"role": "assistant", "content": next_article})

        if next_article == 'back':
            next_article = previous_article_stack.pop()
        else:
            previous_article_stack.append(current_article)
            current_article = next_article
    
    print(f'Used tokens: {used_tokens}')
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

    for pair in ori_dest[:5]:
        print(pair)
        path = play_wikispeedia_game(pair[0], pair[1], G, client)
        print(path)
    




    