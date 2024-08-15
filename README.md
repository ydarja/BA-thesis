This a working repository for my Bachelor thesis of Compuational linguistics.

**Author:** Darja Jepifanova ([@ydarja](https://github.com/ydarja))
**Advisor:** Çağrı Çöltekin ([@coltekin](https://github.com/coltekin))


**Idea:** I am working on creating an evaluation metric of specificity. Specificity measures the amount of detail present in a text. This metric could be useful for tasks like automatic summarization, 
extracting specific information for RAG or active learning. Later I would like to compare my new metric to existing similar metrics and evaluate it on some practical application. For instance, I could take some pre-trained
summarization model and fine-tune it on:

1) randomly picked reference summaries
2) summaries based on my metric
3) summaries based on other metrics

Then we could compare performance of these different versions and conclude if this new metric is of any use...

**Model:**  So how am I going to create this specificity metric? For my training data I am using WordNet lexical database that captures semantic relations between words. Words 
are organized into hierarchies, defined by hypernym or `IS A` relationships. We assume that higher synsets represent more abstract meaning, when lower concepts and leafs tend to be more specific.
Therfore, I create training data by picking pairs of synsets representing different level of specificity (located on different depth in the hierarchy) and extracting their example sentences. Label `0` 
indicates that the first sentence is more specific, and label `1` show that the second sentence is more specific accordingly. Then I train a siamese neural network that encodes both sentences with BERT contextualized 
embeddings and gets specificty score for both of them  using a non-linear transformation followed by a sigmoid function to get a value between 0 and 1. Then these scores are compared and the model outputs
a binary value indicating which sentence is more specific.

**Other resources:**
 - [Literature review](https://docs.google.com/document/d/1wW2RFaqRNMYdH-o2QZU5np9fsbLkJIitnioF6VofpEE/edit)
 - [Model evaluation](https://docs.google.com/document/d/1D959CRIQF49fzi6wlFjMwuK4wJY_lwe7yA5gJEMSWr0/edit)
 - [Overleaf](https://ru.overleaf.com/read/vmhtyhwvxxnq#ee8e13)

**TO DO:**
- [ ] train on sentences with the same context (optional)
- [X] investigate the obtained specificity scores, e.g. correlation with depth and other metrics
- [ ] create data for human evaluation based on the WikiHow dataset and LLM
- [ ] create and distribute a survey
- [ ] draft of the intro section
