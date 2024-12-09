# Readme

## Project Proposal - Rational or emotional: Comparative Study of Human and LLM Pathfinding in the Wikispeedia Game

### Introduction

**Abstract:**

The project aims to explore and compare the strategies of humans and large language models(LLMs) in the Wikipedia game. The goal of the game is to find the shortest paths between two Wikipedia articles. We will investigate whether LLMs are better at finding the shortest paths than humans in terms of efficiency and whether their strategies for finding paths differ significantly from those of humans. Through these analyses, we aim to explore whether LLMs have a deeper or more superficial understanding of semantics compared to humans and whether the LLMs are more perceptual or rational compared to humans.

**Motivation:**

By exploring the differences in semantic understanding and exploration patterns between humans and LLMs, the project helps to understand the cognitive strengths as well as limitations of AI and humans. This will help determine whether AIs have near-human thinking patterns and comprehension abilities. As a result, the project can advance our knowledge and judgment of AI's semantic comprehension capabilities.

**Research Question and Story:**

In this project, we ought to study the following questions:

Do LLMs outperform humans in finding the shortest paths between semantically closely related and unrelated words in the Wikispeedia game? 

How do LLMs’ strategies differ from human strategies when navigating between concepts? 

Do the paths taken by LLMs reflect a deeper or more superficial understanding of the semantic connections between words compared to human paths? 

Are there any biases or noises in the paths taken by humans?

Is there a possibility of AI a priori?

By answering the question above, we can measure the paths from four different perspectives: performance strengths and weaknesses, strategy differences, depth of understanding and bias existence.

### Implement: 

**Pipeline**:

The research process follows this workflow:

![image-pipline](img/pipline.png)

**Database:**

We will only use the database - wikispeedia that has been provided and based on this, we will extract the human navigation paths in the dataset. And we will also use the API to collect batch browsing data of the LLMs. With the combination, we will use the new dataset to analyse and evaluate. 

**Method/Metrics**

0. Preprocessing and Pre-analysis: Perform initial data cleaning and initial data visualization.
1. After balancing cost and performancee, we choose to apply gpt 4o-mini to generate the navigation path made by LLM. Then in the experiment, we will try different prompts, to analysis its impact on the output, like whether there is bias (such as introducing prior knowledge, etc.), and finally choose one to generate the AI navigation paths dataset, with the same distribution of <origin, destination> pairs as human navigation paths dataset. 
2. Graph construction: the articles and the links between them can be naturally structured as digraphs.
3. To measure the difference between the navigation paths made by human and LLM, we could focus on: (i) The statistic of the path, such as the average path length, the most frequently accessed node and its feature, the difference between decisions made by human and LLM like the title level distribution and the title position distribution. (ii) The metrics to measure how closer each move made to the destination, such as the distance change to the destination, or the embedding change which is detailed in 4.
4. If we map each node in the graph into a vector, then we can get a measure of the distance between two nodes, which can be on the graph scale and semantic scale. (i) Graph embedding: by applying Node2Vec, this embedding contains the structural information related to graph. (ii) Semantic embedding: by applying SentenceTransformer, we could turn each node(title) into vector, which contains the semantic information to the document. Once the embedding is got, the distance between two nodes can be implemented as Euclidean distance or cosine similarity, then the efficiency or semantic interpretability for each move can be measured.


**Tool**

Python

OpenAI:Chatgpt 4o-mini

**Timeline**

| Date                 | Content                             |
| :------------------- | ----------------------------------- |
| Now - Nov.15th       | Data preprocessing and pre-analysis |
| Nov.16 - Nov.24th    | LLM dataset generation              |
| Nov.25th -  Dec.8th  | Processing and analysis             |
| Dec.9th - Dec.15th   | Evaluation, comparison and report   |
| Dec.16th - Dec.20th  | Final Check and Supplement          |

### Conclusion

This project compares the abilities of humans and LLMs to find paths in Wikipeedia games, aiming to understand the differences in strategy, efficiency, and semantic understanding. This study will determine whether LLMs exhibit similarities to human reasoning or possess their own unique logical patterns. Ultimately, our goal is to better assess the current capabilities of LLM models in understanding complex information.

## Project Structure

The directory structure of new project looks like this:

```
├── data                        <- Project data files
│
├── src                         <- Source code
│   ├── data                            <- Data directory
│   ├── models                          <- Model directory
│   ├── utils                           <- Utility directory
│   ├── scripts                         <- Shell scripts
│
├── tests                       <- Tests of any kind
│
├── results.ipynb               <- a well-structured notebook showing the results
│
├── .gitignore                  <- List of files ignored by git
├── pip_requirements.txt        <- File for installing python dependencies
└── README.md
```


