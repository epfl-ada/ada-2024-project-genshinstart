### Readme.md 
[database]: https://snap.stanford.edu/data/wikispeedia.html	"wikispeedia"
file containing the detailed project proposal (up to 1000 words). Your README.md should contain:
- Title
- Abstract: A 150 word description of the project idea and goals. What’s the motivation behind your project? What story would you like to tell, and why?
- Research Questions: A list of research questions you would like to address during the project.
- Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. Show us that you’ve read the docs and some examples, and that you have a clear idea on what to expect. Discuss data size and format if relevant. It is your responsibility to check that what you propose is feasible.
- Methods
- Proposed timeline
- Organization within the team: A list of internal milestones up until project Milestone P3.
- Questions for TAs (optional): Add here any questions you have for us related to the proposed project.

Rational or emotional: Comparative Study of Human and LLM Pathfinding in the Wikispeedia Game
Abstract: A 150 word description of the project idea and goals. 
What’s the motivation behind your project? 
The purpose of our project is to prepare the strategies of humans and LLM in navigating the "Wikipedia Game", in which players need to link one article to another one by using the shortest paths. 
What story would you like to tell, and why? 
Research Questions: 
Do LLMs outperform humans in finding shortest paths between semantically closely related and unrelated words in the Wikispeedia game?
How do LLMs’ strategies differ from human strategies when navigating between concepts?
Do the paths taken by LLMs reflect a deeper or more superficial understanding of the semantic connections between words compared to human paths?
Are there any biases or noises in the paths taken by humans?

Proposed additional datasets (if any): List the additional dataset(s) you want to use (if any), and some ideas on how you expect to get, manage, process, and enrich it/them. 

We will not use additional dataset. But we will use LLM to generate a new dataset of navigation paths.

Method: 
1. Prompt the LLM: Describe the game, and give it the inputs: first input the origin and target, and the nodes adjacent to the origin, then after LLM's selection, input the corresponding neighbor, until reach the target. 
2. A measure of reason and sensibility: we can calculate the mean similarity of embeddings in adjacent nodes in a path, to stand for the sensibility. 

Methods Proposed timeline 
Setting Up LLMs to Play the Game: Use popular LLMs to play the Wikispeedia game by feeding the current article and possible links into the model, asking it to predict the best next step to reach the goal.
Human vs. LLM Path Comparison: Compare the length, efficiency, and relevance of paths taken by LLMs versus those taken by humans. Analyze where LLMs diverge from human choices and whether those divergences make sense semantically.
Analysis of Understanding: Investigate whether LLMs display an understanding of concept relationships similar to humans. Determine if the human paths are statistically longer than LLMs, will this reveal any noise in the human behaviors.


Organization within the team: A list of internal milestones up until project Milestone P3. 

Questions for TAs: Add here any questions you have for us related to the proposed project.

Is the research plan feasible or do you think it’s crisp enough?
Do you have any recommendations for quantifying biases or measuring rationality in generated paths?


Example: 
origin: 11_century, target: fire
The path GPT went through: 
1. Starting from the 11th century page, let’s prioritize links that might relate to topics where fire or its history could be discussed, like technology, culture, or warfare.
Looking at this list, I’ll choose:
First choice: "Crusades"
The Crusades involved significant historical events and warfare, which often includes references to fire as a weapon, symbol, or technological element.
Let me know if that link gets me closer!
2. I'll go with "War" next, since fire often appears in the context of warfare, weapons, and strategies.
This choice seems like a strong bet for bringing us closer to the target topic!
3. Next, I’ll select "Weapon" since this topic is closely related to fire in terms of both traditional and modern weaponry. It should lead us closer to discussions about fire!
4. I'll choose "Firearm" next. Firearms directly relate to fire, both in their operation and historical context, making it a logical jump toward the target!
5. Next, I’ll go with "Gunpowder." Gunpowder is a key element in the history of firearms and is closely related to the concept of fire. This should bring us even closer to the target!
6. I'll select "Fire" next, as it is the target article we're aiming for! Let's see if this jump gets us there.





Comments from TA:
Estimation of cost
Greedy
Start from small models
https://huggingface.co/meta-llama/Llama-3.2-1B
https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
Prompt design can matter

Compare strategies:
More hyperlinks in initial steps
Laziness of human choices (tend to make early choices)
Prompts with or without passages to see whether LLMs also have same laziness

https://docs.google.com/document/d/182tSWUMvZoNp5k80JT99ScAzWbSo26JNORv9d36gqtk/edit?usp=sharing
