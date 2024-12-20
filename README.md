# Readme

## Project Who's the Better Navigator? A Race Through Wikipedia City: Humans vs GPT

### Data Story: https://epfl-ada.github.io/ada-2024-project-genshinstart/
### Introduction

**Abstract:**

The project aims to explore and compare the strategies of humans and GPT in the Wikispeedia game. The goal of the game is to find the shortest paths between two Wikipedia articles. We investigated the difference of human navigation paths and GPT navigation paths and there strategy. Besides, by comparing with optimal paths, we tried to improve the prompt for better performance through extracting the features of better paths, and through prompt optimization by applying [textgrad](https://arxiv.org/abs/2406.07496).

**Motivation:**

With the rapid development of LLMs, its application scenarios have become more extensive, whether it can migrate to the search problem is an interesting topic. In the Wikispeedia game, finding the possible shortest paths in a graph under the implicit semantic information between nodes could be an example to this problem.

**Research Question:**

What are the differences between AI, human and optimal paths and strategies?  
Could we improve the prompt to get better results?

### Implement: 

**Database:**

We used wikispeedia that has been provided and based on this, we extracted the human navigation paths in the dataset. Besides, we generated paths for 159 most frequent (start article, destination) pairs in human navigation paths using GPT 4o-mini, which can be downloaded from https://drive.google.com/file/d/19cHamrMGMWwMOOKIl_zdv7Opp29B_gbG/view?usp=drive_link, to see more details related to these generated paths, please visit our website.
This file contains
- paths: The paths generated using baseline prompt.
- paths_improved_prompt: The paths generated using manually improved prompt.
- paths_before_tuning: The paths generated under modified task.
- paths_after_tuning: The paths generated using prompt tuned by GPT-4o.
- paths_after_tuning_4o_mini: The paths generated using prompt tuned by GPT-4o mini.

**Method/Metrics**

1. Path generation: after balancing cost and performance, we choose to apply gpt 4o-mini to generate the navigation path. To see more details about the GPT and prompt settings, please visit our website.
2. Metrics to measure difference between paths: 
- The statistic of the path, such as the average path length. 
- The semantic efficiency for each movement: we introduced clossness score, which is the difference between distance to destination before and after one movement. The distance here is cosine similarity between sentences embeddings of article titles, which reflects semantic distance between articles.
- Out degree of chosen article: This could reflect the 'potential' for a chosen article. We also analysed neighbor preference condition on all the neighbors has low semantic similarities with the destination, to get insight for better strategy.
3. Prompt improvement: 
- Manually improvement: Based on the discovery above, we provide some hints for the GPT-4o mini model, and the generated paths actually get improved.
- Prompt optimization based on textgrad: We redefined the task for applying the textgrad pipeline, generate training samples and get prediction from GPT-4o mini model, then use feedback provided by GPT-4o to improve the prompt.

**File Description**

| File/Directory Name         | Description                                                                 |
| --------------------------- | --------------------------------------------------------------------------- |
| `data/`                     | Directory containing the raw and processed data files.                      |
| `src/`                      | Directory containing the source code for the project.                       |
| `comparison.ipynb`          | The main results for our project. |
| `generate.py`          | Script for generating navigation paths using GPT-4o mini, under the initial task. |
| `textgrad.ipynb`          | Prompt optimization using textgrad. |
| `README.md`                 | This README file.                                                           |

Because generating the paths using GPT costs money, we didn't include and rerun it in comparison.ipynb. To generate the paths under the initial task, you can use generate.py, and for the prompt optimization process, you can see textgrad.ipynb. Alternatively, you can simply download the generated paths from this link https://drive.google.com/file/d/19cHamrMGMWwMOOKIl_zdv7Opp29B_gbG/view?usp=drive_link or ./data/ folder, then you can run all the analysis in comparison.ipynb based on these files

**Install the dependency**
```bash
pip install -r requirements.txt
```

**Tool**

Python

OpenAI:GPT-4o and GPT-4o mini

[textgrad](https://arxiv.org/abs/2406.07496)
       

### Conclusion

This project compares the abilities of humans and GPT to find paths in Wikispeedia games, aiming to understand the differences in strategy, efficiency, and semantic understanding. We also tried prompt optimization based on TextGrad, which could potentially be generalized to solve search problems using LLMs.

### Contribution
- Yexiang Cheng: Preliminary data analysis, GPT paths generation, Paths comparation, Prompt optimization using textgrad and analysis, Result explaination.


