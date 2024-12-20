import urllib
import textgrad as tg
import random
import networkx as nx

def generate_path(start_article, target_article, graph, model):
    '''
    Generate a path from the start article to the target article using modifed task by gpt-4o-mini
    Parameters:
    - start_article: str, the start article
    - target_article: str, the target article
    - graph: nx.DiGraph, the graph
    - model: tg.BlackboxLLM, the model

    Returns:
    - path: list, the path from the start article to the target article
    '''

    path = [urllib.parse.unquote(start_article)]
    cnt = 0
    while start_article != target_article:
        cnt += 1
        if cnt > 20:
            print(path)
            return ['No path found']
        
        # get the neighbors of the current article
        neighbors = list(graph.successors(start_article))
        neighbors = [urllib.parse.unquote(neighbor) for neighbor in neighbors]

        # remove the neighbors that are already in the path, to avoid cycles
        neighbors = [neighbor for neighbor in neighbors if neighbor not in path]

        # ask the model to choose the next article
        question = f'You are currently at the article "{urllib.parse.unquote(start_article)}". The target article is "{urllib.parse.unquote(target_article)}". The links you can choose from are: {', '.join(neighbors)}.'
        question = tg.Variable(question, role_description="Navigation query", requires_grad=False)
        prediction = model(question)

        # process the prediction
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
    '''
    Generate a question and answer pair for the path finding task
    
    Parameters:
    - graph: nx.DiGraph, the graph
    
    Returns:
    - question: str, the question
    - answer: str, the answer
    - neighbors: list, the neighbors of the current article
    - next_articles: list, the next articles in the shortest path
    '''

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
    
    # get all the neighbors of the current article
    neighbors = list(graph.successors(current_article))
    neighbors = [urllib.parse.unquote(neighbor) for neighbor in neighbors]

    # get the next articles in the shortest path
    next_articles = set([urllib.parse.unquote(next_article) for next_article in next_articles])
    current_article = urllib.parse.unquote(current_article)
    target_article = urllib.parse.unquote(target_article)
    question = f'You are currently at the article "{current_article}". The target article is "{target_article}". The links you can choose from are: {', '.join(neighbors)}.'
    answer = f'The next article in the shortest path from "{current_article}" to "{target_article}" is in "{', '.join(next_articles)}".'
    return question, answer, neighbors, next_articles

