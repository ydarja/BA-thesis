# Creating and Evaluating a New Specificity Metric based on WordNet

**Author:** Darja Jepifanova ([@ydarja](https://github.com/ydarja))
**Advisor:** Çağrı Çöltekin ([@coltekin](https://github.com/coltekin))

**Project Overview:**  
This repository contains the code, data, and resources for my  bachelor thesis 
on developing and evaluatinga new specificity metric based on WordNet. 

**Abstract:**  
In computational linguistics, specificity quantifies how much detail is engaged in
text. It is useful in many NLP applications such as summarization and information
extraction. It can also be used as a text quality metric, which can provide a more
transparent evaluation of machine-generated text. Yet to date, expert-annotated
data for sentence-level specificity are scarce and confined to the news or social
media genre. In addition, systems that predict sentence specificity are classifiers
trained to produce binary labels (general or specific).
In this thesis, I introduce a new specificity metric based on WordNet, which posits
that lower synsets in the semantic hierarchy represent more specific concepts.
Based on this principle, I trained a Siamese network to distinguish between specific
and general sentences based on the depth of the corresponding synsets. The
evaluation of the resulting continuous specificity scores involved statistical analysis,
comparisons with existing metrics, and human evaluations. The analysis revealed
a Pearson correlation of 0.38 with a Twitter-based dataset containing annotated
specificity, suggesting promising outcomes. However, human evaluations resulted
in a correlation of just 0.19 and low inter-annotator agreement, highlighting the
metric’s limitations. Despite these challenges, this study provides a foundation for
future enhancements and applications in natural language generation, aiming to
improve the quality and precision of machine-generated texts.

*For questions or comments, feel free to contact me via email: ydarja@gmail.com*

