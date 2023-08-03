# Synthetic Multi-turn Dialogue Generation with Iterative Self-Refinement

In-context learning for initial response generation, response self-refinement for improved specificity, and new user query generation, in the document-grounded setting. Currently includes the MultiDoc2Dial and AskHR datasets, and can be further extended to other content-grounded datasets, given they follow a standardized format consistent with the current datasets. 

# Installation and Setup

First, create a new conda environment, as follows:

```sh
conda create -y -p ./{env-name} python=3.10
conda activate {env-name}
```

Next, install the package requirements as provided in requirements.txt:

```sh
pip install -r requirements.txt 
```

Querying models via BAM requires an API key, which can be obtained at https://bam.res.ibm.com/. Save this key in a file titled ```.env``` as follows:

```sh
GENAI_KEY={your-api-key}
```

Finally, to set the PYTHONPATH, run the following command:

```sh
source setup.sh
```

# Single-turn Response Generation

To run single-turn response generation with iterative refinement, use the following command:
```sh
bash scripts/run_response_gen.sh
```

In ```scripts/run_response_gen.sh```, the dataset, model to be queried via BAM, the number of samples in the dataset to run, and the maximum number of refinement attempts can be modified.
Soon to be added: 'hugging-face' as a model source in addition to 'ibm-generative-ai'. 

# Multi-turn Synthetic Dialogue Generation

Observe that the only difference between the shell scripts for the following datasets is the dataset path -- hence, this can be adapted to other content-grounded datasets.

### MultiDoc2Dial

To run multi-turn dialogue generation with iterative refinement on MultiDoc2Dial:

```sh
bash scripts/run\_sdg\_md2d.sh 
```

### AskHR

To run multi-turn dialogue generation with iterative refinement on AskHR:

```sh
bash scripts/run\_sdg\_askhr.sh
```
