# DIANA: a knowledge-driven framework for adaptive data-centric AI 

DIANA is a framework for data-centric AI to support data exploration and preparation.

The central component of DIANA is a knowledge base that collects evidence related to the impact of errors and the effectiveness of data preparation actions, along with the type of input data and the considered machine learning model.

The knowledge base main functionality is to suggest an optimal data preparation pipeline to obtain valuable analysis results. 
This repository contains:

1. The knowledge base components and suggestion mechanism of DIANA.
2. The validation experiments conducted for validating the suggestion mechanism.

---

## Components

The knowledge base (kb) system is composed of 3 predictors and an unreliability score:

1. **kb-ranking**: it contains the information for suggesting the order in which the DQ dimensions should be improved, the trained regression models, their performance and evaluation metrics.
   - Folder: KB_Dimensions
2. **kb-accuracy**: it contains the information for suggesting the optimal outlier detection techniques for improving the accuracy dimension (only for numerical columns), the trained regression models, their performance evaluation metrics.
   - Folder: KB_Accuracy
3. **kb-completeness**: it contains the information for suggesting the optimal data imputation techniques for improving the completeness dimension (for numerical/categorical columns), the trained regression models, their performance evaluation metrics.
   - Folders: KB_Completeness_Num, KB_Completeness_Cat
4. **kb-unreliability**: it contains the unrealiability scores of the predictions extracted for all tested combinations of datasets-models.
   - Folders: KBreliability (all), KBreliability (ranking), KBreliability (completeness), KBreliability (accuracy)

The kb folders listed above contains:

- kb: the kb content including datasets, data profiles, and the results of the kb enrichment process
- notebooks/scripts: the code for traning and testing the kb predictor
- notebooks: for visualizing and analyzing the main results on the computed performance and evaluation metrics of the predictors
- results: the experiments results (performance and evaluation metrics after training the models)

---

## Validation experiments

The validation experiments are composed of:

1. **polluted datasets validation**: experiments validation with systematically polluted datasets with controlled percentage of injected errors.
   - Folders: validation_character, validation_consumer, validation_galaxy, validation_heart, validation_pet, validation_weather
2. **real-world datasets validation**: experiments validation with datasets already containing real-world errors.
   - Folders: running_example (mobile price dataset), running_example_2 (customer satisfaction dataset)

The experiments folders listed above contains:

- dataset: the dataset used for the experiments
- kb: the kb predictors
- scripts: the code for implementing and executing the experiment
- results: the experiments results, the code for visualizing them (plots_A, plots_C), plus additional notebooks with results analysis
- plot: different visualizations of the obtained results
- notebooks: additional notebooks

---

## Installation & Usage

### Prerequisites
Ensure you have the following installed on your system:
- python 3.11 or higher
- pip (python package manager)

### Setup
1. Clone this repository:
   ```bash
   git clone https://github.com/camillasancricca/diana-repo.git
   cd diana-repo
   ```
2. Install the requiements:
   ```bash
   pip install -r requirements.txt
   ```
