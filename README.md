### Topic introduction

The project Generative AI Powered for Programming Education was realized under the guidance of professors teaching the course Generative AI from Universit&auml;t des Saarlandes.

The purpose of the project is to generate hints and program corrections which could help a learner understand the mistakes and bug(s) in their code. 

### Metrics to evaluate a program repair

The repair of a program is evaluated using two metrics
- **RPass**, binary variable, representing the number of passed test cases.
- **REdit**, real and non-negative number, capturing the distance between the buggy and the repaired program. The smallest the different, the better the repair.

### Metrics to evaluate a hint

The quality of a hint is evaluated using 4 separate metrics
- **HCorrect**, binary variable. It is 1 if the generated hint provides correct information for resolving the buggy program.
- **HInformative**, binary variable. It is 1 if the hint provides useful information to help the learner resolve bug(s).
- **HComprehensible**, binary variable. It is 1 if the generated hint is easy to understand, if it is presented in a readable format and if it does not contain any redundant information.
- **HGood**, binary variable. It measure the overall quality of a generated hint. It is 1 if all other variables have the value 1. 

### Usability metrics

In order to account for the memory and time used by each different technique, the following metrics have been used:
- **TrainingTime**. It captures the time used during the fine-tuning process. 
- **TrainingMemory**. It captures the memory used during the fine-tuning process. 

### Project parts

The project was divided into 2 parts:
- Part 1 focused on evaluating the baseline models and exploring prompt engineering techniques for program repair and hint generation.
- Part 2 focused on fine-tuning language models for program repair and hint generation. In addition, exploration of parameter-efficient training with Low-Rank Adaption (LoRA) and multi-task fine-tuning was done.  

### Sources used

- Alkis Gotovos Nachiket Kotalwar and Adish Singla. Hints-In-Browser: Benchmarking Language
Models for Programming Feedback Generation. In NeurIPS (Datasets and BenchmarksTrack), 2024. Paper Link: https://openreview.net/pdf?id=JRMSC08gSF