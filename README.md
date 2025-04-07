<h1 align="center">GPT-Thief: Testing Robustness of Homomorphically Encrypted Split Model LLMs <a href="https://github.com/TrustworthyComputing/gpt-thief/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg"></a> </h1>

## Overview
Large language models (LLMs) are revolutionizing industries by enabling advanced applications such as content generation, customer service agents, and data analysis. These models are often hosted on remote servers to protect intellectual property (IP), raising concerns about the privacy of input data. Fully Homomorphic Encryption (FHE) has been proposed as a solution to ensure privacy during computations on encrypted data. Practical implementations rely on a split model approach, where encrypted data is processed partially on the server and partially on the user's machine. While this method aims to balance privacy and model IP protection, we demonstrate a novel attack vector that enables users to extract the neural network model IP from the server, undermining claimed protections for encrypted computations.


## Usage
The tools and scripts provided in this repository allow users to replicate key findings from the paper, investigate vulnerabilities in split model LLMs, and investigate potential mitigation techniques. 

### How to cite this work
This work has been presented at the 2025 Design Automation and Test in Europe (DATE) conference. The preprint can be accessed [here](https://eprint.iacr.org/2024/1675); you can cite this work as follows:
```
@misc{folkerts2024testing,
      author = {Lars Wolfgang Folkerts and Nektarios Georgios Tsoutsos},
      title = {Testing Robustness of Homomorphically Encrypted Split Model {LLMs}},
      howpublished = {Cryptology {ePrint} Archive, Paper 2024/1675},
      year = {2024},
      url = {https://eprint.iacr.org/2024/1675}
}
```

## Acknowledgments
This work was supported by the National Science Foundation (Award #2239334).

<p align="center">
    <img src="./logos/twc.png" height="20%" width="20%">
</p>
<h4 align="center">Trustworthy Computing Group</h4>
