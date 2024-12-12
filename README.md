# CaLQuest.PT: Towards the Collection and Evaluation of Natural Causal Ladder Questions in Portuguese for AI Agents

This repository contains all codes and data used on our research paper titled "**CaLQuest.PT: Towards the Collection and Evaluation of Natural Causal Ladder Questions in Portuguese for AI Agents**", by "Uriel Lasheras" and "Vládia Pinheiro".
The code available in this repository contain all methods needed to reproduce our results, includind downloading data from sources, preporcessing, generating liguistic evaluation metrics, making inferences using LLMs (On this paper, GPT-4o) and generating inference evaluation metrics.

### Credits

Part of the code available on "./extractos/utils.py", "./extractos/sharegpt.py" and "./extractos/wildchat.py" are modified versions of [CausalQuest](https://github.com/roberto-ceraolo/causal-quest) code, modified for using the data of those datasets in Portuguese.

### Dataset

All datasets used on the paper for Linguistic Metrics analysis and LLM Inference and Metric analysis are available on the [experiment_data](experiment-data) folder.

Original Datasets are not directly available, for exception of Reddit dataset which is available as an anonymized version by Reddit's staff request. This datast specificaly is available due to future runs having updated data and may lose many of the questions used originaly. So, for reproducing our results, this dataset will provide all Reddit questions used on the paper.

Other datasets available on this folder are our results on full dataset and Golden Collection.

### Dataset and Data Licenses

The following are the licences for each of the sources used in the dataset:
- **ShareGPT**: Apache-2, available at [https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered).
- **WildChat**: AI2 ImpACT License – Low Risk Artifacts, available at [https://allenai.org/licenses/impact-lr](https://allenai.org/licenses/impact-lr)
- **Reddit**: Unknown Free to use for Non-Commercial (Academmic) purposes.

We invite all users of CausalQuest to carefully read and respect the licenses of the sources.

## Repository structure

The following is the structure of the repository:

- **data_gen**
    - *empty*: Will be used for storing generated data

- **metrics_gen**
    - *empty*: Will be used for storing generated metrics

- **experiment_data**
    - reddit_pr_br_lg.xlsx - Reddit Data (Anonymized)
    - unified_ptbr_gpt_SAMPLE_RD.xlsx - Golden Collection Data from Reddit source, with inferences for Causality.
    - unified_ptbr_gpt_SAMPLE_RD2.xlsx - Golden Collection Data from Reddit source, with inferences for Causality and Action Class.
    - unified_ptbr_gpt_SAMPLE_RD3.xlsx - Golden Collection Data from Reddit source, with inferences for Causality, Action Class and Pearl Class.
    - unified_ptbr_gpt_SAMPLE_RD_CoT.xlsx - Golden Collection Data from Reddit source, with inferences for Causality. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_RD2_CoT.xlsx - Golden Collection Data from Reddit source, with inferences for Causality and Action Class. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_RD3_CoT.xlsx - Golden Collection Data from Reddit source, with inferences for Causality, Action Class and Pearl Class. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_WC.xlsx - Golden Collection Data from WildChat source, with inferences for Causality.
    - unified_ptbr_gpt_SAMPLE_WC2.xlsx - Golden Collection Data from WildChat source, with inferences for Causality and Action Class.
    - unified_ptbr_gpt_SAMPLE_WC3.xlsx - Golden Collection Data from WildChat source, with inferences for Causality, Action Class and Pearl Class.
    - unified_ptbr_gpt_SAMPLE_WC_CoT.xlsx - Golden Collection Data from WildChat source, with inferences for Causality. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_WC2_CoT.xlsx - Golden Collection Data from WildChat source, with inferences for Causality and Action Class. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_WC3_CoT.xlsx - Golden Collection Data from WildChat source, with inferences for Causality, Action Class and Pearl Class. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_SG.xlsx - Golden Collection Data from ShareGPT source, with inferences for Causality.
    - unified_ptbr_gpt_SAMPLE_SG2.xlsx - Golden Collection Data from ShareGPT source, with inferences for Causality and Action Class.
    - unified_ptbr_gpt_SAMPLE_SG3.xlsx - Golden Collection Data from ShareGPT source, with inferences for Causality, Action Class and Pearl Class.
    - unified_ptbr_gpt_SAMPLE_SG_CoT.xlsx - Golden Collection Data from ShareGPT source, with inferences for Causality. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_SG2_CoT.xlsx - Golden Collection Data from ShareGPT source, with inferences for Causality and Action Class. Inferred using CoT.
    - unified_ptbr_gpt_SAMPLE_SG3_CoT.xlsx - Golden Collection Data from ShareGPT source, with inferences for Causality, Action Class and Pearl Class. Inferred using CoT.

- **prompts**
    - few-shot_causality.txt - Few-Shot prompts for inference on Causality Axis.
    - few-shot_actions.txt - Few-Shot prompts for inference on Action Axis.
    - few-shot_pearl.txt - Few-Shot prompts for inference on Pearl Axis.
    - cot_causality.txt - Chain of Thought prompts for inference on Causality Axis.
    - cot_actions.txt - Chain of Thought prompts for inference on Action Axis.
    - cot_pearl.txt - Chain of Thought prompts for inference on Pearl Axis.

- **extractors**
    - reddit.py
    - wildchat.py
    - sharegpt.py
    - utils.py

- **inference**
    - inference_fs.py
    - inference_cot.py
    - questions_5w2h.py

- **metrics**
    - inference.py
    - linguistic.py

## Code Setup

Clone the repository using the following command:

    git clone <path>
    cd CalQuest_PT

Then, install the dependencies:

    pip install -r requirements.txt

## Code Usage
The following command will then generate the full dataset:

`python main.py`
    