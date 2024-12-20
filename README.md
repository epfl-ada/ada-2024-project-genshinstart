# Who's the Better Navigator? A Race Through Wikipedia City: Humans vs GPT

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

We used wikispeedia that has been provided and based on this, we extracted the human navigation paths in the dataset. Besides, we generated paths for 159 most frequent (start article, destination) pairs in human navigation paths using GPT 4o-mini, which can be downloaded from [Google Drive](https://drive.google.com/file/d/19cHamrMGMWwMOOKIl_zdv7Opp29B_gbG/view?usp=drive_link), to see more details related to these generated paths, please visit our [Story website](https://epfl-ada.github.io/ada-2024-project-genshinstart/).
This file contains
- paths: The paths generated using baseline prompt.
- paths_improved_prompt: The paths generated using manually improved prompt.
- paths_before_tuning: The paths generated under modified task.
- paths_after_tuning: The paths generated using prompt tuned by GPT-4o.
- paths_after_tuning_4o_mini: The paths generated using prompt tuned by GPT-4o mini.

**Method/Metrics**

1. Path generation: after balancing cost and performance, we choose to apply gpt 4o-mini to generate the navigation path. 
2. Metrics to measure difference between paths: 
- Path Statistics: Metrics such as the average path length can directly show the differences between paths.
- The semantic efficiency for each movement: We introduced a closeness score, which is the difference in distance to the destination before and after one movement. The distance here is measured using cosine similarity between sentence embeddings of article titles, reflecting the semantic distance between articles. This metric provides a way to analyze how much semantically closer each movement brings you to the destination, which helps explore the semantic meaning in different paths.
- Out degree of chosen article: This metric reflects the 'potential' of a chosen article. We also prefered neighbors' degree, conditioning on all neighbors with low semantic similarities to the destination, to gain insights for better strategy.
3. Prompt improvement: 
- Manually improvement: Based on the discoveries above, we provided some hints for the GPT-4o mini model, and the generated paths actually improved.
- Prompt optimization based on textgrad: We redefined the task for applying the TextGrad pipeline, generated training samples, and obtained predictions from the GPT-4o mini model. We then used feedback provided by GPT-4o to improve the prompt. After several rounds, we got the improved prompt.

See our [Story](https://epfl-ada.github.io/ada-2024-project-genshinstart/) for more details.

**File Description**

| File/Directory Name         | Description                                                                 |
| --------------------------- | --------------------------------------------------------------------------- |
| `data/`                     | Directory containing data files, including wikispeedia paths and GPT generated paths.                      |
| `src/`                      | Directory containing the source code for the project                       |
| `results.ipynb`          | The main results for our project. |
| `generate.py`          | Script for generating navigation paths using GPT-4o mini, under the initial task setting. |
| `textgrad.ipynb`          | Prompt optimization using textgrad, under the modified task setting |
| `README.md`                 | This README file.                                                           |

Because generating the paths using GPT costs money, we didn't include and rerun it in results.ipynb. To generate the paths under the initial task, you can use generate.py, and for the prompt optimization process, you can see textgrad.ipynb. Alternatively, you can simply download the generated paths from [Google Drive](https://drive.google.com/file/d/19cHamrMGMWwMOOKIl_zdv7Opp29B_gbG/view?usp=drive_link) or ./data/ folder, then you can run all the analysis in results.ipynb based on these files

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

### References

- Mert Yuksekgonul, Federico Bianchi, Joseph Boen, Sheng Liu, Zhi Huang, Carlos Guestrin, and James Zou. "TextGrad: Automatic 'Differentiation' via Text." arXiv preprint arXiv:2406.07496, 2024. [https://arxiv.org/abs/2406.07496](https://arxiv.org/abs/2406.07496)

### Contribution
- Yexiang Cheng: Preliminary data analysis, GPT paths generation, paths comparison, prompt optimization using textgrad and analysis, result explaination.
- Rongxiao Qu: Problem formulation, preliminary data analysis, paths comparison, plotting graphs during analysis, result explaination.
- Zhengping Qiao: Preliminary data analysis, designing and development the data-story webpage, also contributed to the story content.
- Tian Jing: Writing up the proposal and the data story.
- Zhiyao Yan: Preliminary data analysis, paths comparison, integrate webpage and results report.

