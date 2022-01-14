### Predicting Text Difficulty
This projects shows an attempt to create a model that would predict text difficulty in a statement or sentence with high accuracy. For each statement or sentence, the model would predict whether it needs to be simplified or not simplified. This could potentially help build a tool that could aid editors, who simplify text and increase its readability of wikipedia pages for those who have learning/reading difficulties or have English as a second language.

### Files
- models_run.py 
- eval_model.py
- EDA_FeatureSelection_Evaluation.ipynb

### Preparation
The code requires pandas, numpy, spacy, scikit-learn, matplotlib and re to run. 

More details about the project can be found in this blog post [here](https://jessjkim-1.medium.com/predicting-text-difficulty-b07f64b8a439?source=friends_link&sk=beeedbe708690fe63979b02bd8aedb77).

### References
Cdimascio. (n.d.). Cdimascio/py-readability-metrics: 📗 score text readability using a number of formulas: Flesch-Kincaid grade level, Gunning Fog, Ari, Dale Chall, smog, and more. GitHub. Retrieved November 28, 2021, from https://github.com/cdimascio/py-readability-metrics.

Collins-Thompson, K., & Callan J. P. (2004). A Language Modeling Approach to Predicting Reading Difficulty. Proceedings of the Human Language Technology Conference of the North American Chapter of the Association for Computational Linguistics: HLT-NAACL 2004. Retrieved November 28, 2021, from https://aclanthology.org/N04-1025/

Sha, H., & Yep, T. (2018). CS 229 Project Report: Text Complexity (Natural Language). CS 229 projects. Retrieved November 28, 2021, from https://cs229.stanford.edu/proj2018/report/185.pdf. 
