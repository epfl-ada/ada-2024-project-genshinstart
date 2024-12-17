import urllib
import textgrad as tg
import random
import networkx as nx

def generate_path(start_article, target_article, graph, model):
    path = [urllib.parse.unquote(start_article)]
    cnt = 0
    while start_article != target_article:
        cnt += 1
        if cnt > 20:
            print(path)
            return ['No path found']
        neighbors = list(graph.successors(start_article))
        neighbors = [urllib.parse.unquote(neighbor) for neighbor in neighbors]

        neighbors = [neighbor for neighbor in neighbors if neighbor not in path]
        question = f'You are currently at the article "{urllib.parse.unquote(start_article)}". The target article is "{urllib.parse.unquote(target_article)}". The links you can choose from are: {', '.join(neighbors)}.'
        question = tg.Variable(question, role_description="Navigation query", requires_grad=False)
        prediction = model(question)
        choice = str(prediction).split(":")[1].split("\n")[0].strip()
        if '"' in choice:
            choice = choice[1:-1]
        if choice not in neighbors:
            print(f"neighbors: {neighbors}\nchoice: {choice}")
            return ['Invalid choice']
        path.append(choice)
        start_article = urllib.parse.quote(choice)
    return path

def get_question_and_answer(graph):
    # randomly choose a current article and a target article
    current_article = random.choice(list(graph.nodes))
    target_article = random.choice(list(graph.nodes))
    while not nx.has_path(graph, source=current_article, target=target_article):
        target_article = random.choice(list(graph.nodes))

    # get all the shortest paths
    shortest_paths = nx.all_shortest_paths(graph, source=current_article, target=target_article)
    next_articles = []
    for path in shortest_paths:
        next_articles.append(path[1])
    
    neighbors = list(graph.successors(current_article))
    neighbors = [urllib.parse.unquote(neighbor) for neighbor in neighbors]
    next_articles = set([urllib.parse.unquote(next_article) for next_article in next_articles])
    current_article = urllib.parse.unquote(current_article)
    target_article = urllib.parse.unquote(target_article)
    question = f'You are currently at the article "{current_article}". The target article is "{target_article}". The links you can choose from are: {', '.join(neighbors)}.'
    answer = f'The next article in the shortest path from "{current_article}" to "{target_article}" is in "{', '.join(next_articles)}".'
    return question, answer, neighbors, next_articles

