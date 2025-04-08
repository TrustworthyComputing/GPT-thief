## Overview
To run the code, download the datasets from [here](https://drive.google.com/drive/folders/1QcqJQFfXxIG-K-NIz-OdayKuwtLE-cvz?usp=sharing) and place them in the ```data/``` directory. These datasets represent the input queries and corresponding outputs of the FHE-encrypted attention head. 
For training the shadow model based on the collected dataset, run:
```
python3 thief.py
```
to reconstruct the GPT attention head.

## Concrete-ML
We provide the original Jupyter notebook for the vulnerable FHE-encrypted GPT model. Our work focused on the SingleAttentionHead setup.
The full Concrete-ML LLM library can be found [here](https://github.com/zama-ai/concrete-ml/tree/main/use_case_examples/llm).
