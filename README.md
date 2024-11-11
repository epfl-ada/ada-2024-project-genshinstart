## Readme

### Project Proposal - Rational or emotional: Comparative Study of Human and LLM Pathfinding in the Wikispeedia Game

#### Introduction

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

By answering the question above, we can measure the paths from four different perspectives: performance strengths and weaknesses, strategy differences, depth of understanding and bias existence.

#### Implement: 

**Pipeline**:

The research process follows this workflow:

![image-pipline](img/pipline.png)

**Database:**

We will only use the database - wikispeedia that has been provided and based on this, we will extract the human navigation paths in the dataset. And we will use the API to collect batch browsing data of the LLMs.

**Method/Matrix**

0. Preprocessing and Pre-analysis: Perform initial data cleaning and initial data visualization.

1. we will generate and visualize graph node embeddings using Node2Vec, enabling analysis of node relationships in a reduced-dimensional space.
2. We will design a method based on graph embedding closeness score to judge the merit of the method, which shows a clear advantage in the optimal path

**Timeline**

| Date                 | Content                             |
| :------------------- | ----------------------------------- |
| Now - Nov.15th       | Data preprocessing and pre-analysis |
| Now - Nov.23rd       | API exploration                     |
| Nov.16th - Nov.30th  | Dataset generation                  |
| Nov.30th -  Dec.13rd | Processing and analysis             |
| Dec.13th - Dec.20th  | Evaluation, comparison and report   |

**Conclusion**

This project compares the abilities of humans and LLMs to find paths in Wikipeedia games, aiming to understand the differences in strategy, efficiency, and semantic understanding. This study will determine whether LLMs exhibit similarities to human reasoning or possess their own unique logical patterns. Ultimately, our goal is to better assess the current capabilities of LLM models in understanding complex information.

