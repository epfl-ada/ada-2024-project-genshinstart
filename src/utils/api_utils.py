from openai import OpenAI

def choose_next_article(current_article, possible_links, target_article, visited_articles, client, model):
    
    # Define the prompt
    prompt = f"""
    You are playing a game where you need to navigate from one Wikipedia article to the target article in the fewest steps possible.

    Current article: {current_article}
    Target article: {target_article}

    Articles you have previously visited: {', '.join(visited_articles) if visited_articles else "No article has been visited"}
    
    You can move to one of the following articles:
    {', '.join(possible_links)}

    Or if you think returning to the previous article is beneficial, you may do so. 
    
    Reply only with the name of the article you want to move to, with no explanation or marks.
    """

    # Make the updated API call
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert in navigating Wikipedia topics."},
            {"role": "user", "content": prompt}
        ]
    )

    # Return the chosen article and token usage
    return response.choices[0].message.content, response.usage.total_tokens



def play_wikispeedia_game(start_article, target_article, link_map, client, model="gpt-4"):
    current_article = start_article
    visited_articles = set()  # Track visited articles
    steps = 0
    total_tokens = 0

    while current_article != target_article:
        # Check if the article has outgoing links
        if current_article not in link_map:
            print(f"No outgoing links from {current_article}. Game over!")
            print(f"Total token usage: {total_tokens}")
            return

        # Get possible next articles
        possible_links = link_map[current_article]
        visited_articles_list = list(visited_articles)

        # Call GPT to get the next article
        next_article, token_usage = choose_next_article(current_article, possible_links, target_article, visited_articles_list, client=client, model=model)
        total_tokens += token_usage
        print(f"Step {steps}: Moved from '{current_article}' to '{next_article}'")

        # If next_article is the target, the game ends
        if next_article == target_article:
            print(f"Reached the target article '{target_article}' in {steps + 1} steps!")
            print(f"Total token usage: {total_tokens}")
            return

        # Add the current article to visited
        visited_articles.add(current_article)

        # Move to the next article
        current_article = next_article
        steps += 1

        # Limit steps to prevent infinite loops or excessive token usage
        if steps > 15:
            print("Step limit reached. Game terminated.")
            print(f"Total token usage: {total_tokens}")
            return

    # print(f"Reached the target article '{target_article}' in {steps} steps!")