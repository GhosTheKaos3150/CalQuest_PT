import os
import openai
import pandas as pd
from tqdm import tqdm

def generate_inference(golden_collection: pd.DataFrame):
    '''
    This function generates the Inference for all classifications on the three-axis classification used on the
    CalQuest.PT (Causality, Action Class and Pearl Class). Prompts are available on the "./prompts/" folder as
    .txt files. This function will use Few-Shot with Chain of Thought method.
    
    Parameters:
    : golden_collection - Golden Collection dataset, before inference.
    
    Return: Golden Collection, updated with Chain of Thought results.
    '''
    
    gc_infered_chain_of_thought = inference_causality(golden_collection)
    gc_infered_chain_of_thought = inference_actions(gc_infered_chain_of_thought)
    gc_infered_chain_of_thought = inference_pearl(gc_infered_chain_of_thought)
    
    return gc_infered_chain_of_thought

def inference_causality(golden_collection: pd.DataFrame):
    '''
    This function generates inference for Causality classification.
    
    Parameters:
    : golden_collection - Golden Collection dataset, before inference.
    
    Return: Golden Collection, updated with Chain of Thought results for Causality classification.
    '''
    
    api_key = os.environ['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_key)
    
    name_ref = {'reddit': 'RD', 'wildchat': 'WC', 'sharegpt': 'SG'}
    
    prompt_causal = get_prompt_cot_causality()
    
    if not os.path.exists("./data_gen/unified_ptbr_gpt_GC_CoT.xlsx"):
        for i, sample in tqdm(golden_collection.iterrows()):
            try:
                chat_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages = [
                        {"role": "system", "content": prompt_causal.format(PERGUNTA=sample["question"])},
                    ],
                )
            except Exception as e:
                print(e)
                golden_collection.at[i, 'is_causal'] = False
                golden_collection.at[i, 'causal_reasoning'] = ''
                continue

            response = chat_completion.choices[0].message.content
            reasoning = response.split("RACIOCÍNIO:")[-1].strip()
            classification = response.split("RACIOCÍNIO:")[0].split("CATEGORIA:")[-1].strip()
            
            golden_collection.at[i, 'is_causal'] = True if classification == "Causal" or classification == "<Causal>" else False
            golden_collection.at[i, 'causal_reasoning'] = reasoning
        
        golden_collection.to_excel("./data_gen/unified_ptbr_gpt_GC_CoT.xlsx", index=False)
        
        for source in ['reddit', 'wildchat', 'sharegpt']:
            golden_collection[golden_collection['source'] == source].to_excel(f"./data_gen/unified_ptbr_gpt_GC_{name_ref[source]}_CoT.xlsx", index=False)
        
    else:
        golden_collection = pd.read_excel("./data_gen/unified_ptbr_gpt_GC_CoT.xlsx")
        golden_collection.fillna({
            'hr_is_causal': '',
            'hr_action_class': '',
            'hr_pearl_class': '',
            'action_class': '',
            'ac_reasoning': '',
            'pearl_class': '',
            'pc_reasoning': ''
            }, inplace=True)

    client.close()
    
    return golden_collection

def inference_actions(golden_collection: pd.DataFrame):
    '''
    This function generates inference for Actions classification.
    
    Parameters:
    : golden_collection - Golden Collection dataset, before inference.
    
    Return: Golden Collection, updated with Chain of Thought results for Action classification.
    '''
    
    api_key = os.environ['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_key)
    
    name_ref = {'reddit': 'RD', 'wildchat': 'WC', 'sharegpt': 'SG'}
    
    prompt_action = get_prompt_cot_actions()
    
    if not os.path.exists("./data_gen/unified_ptbr_gpt_GC2_CoT.xlsx"):
        for i, sample in tqdm(golden_collection.iterrows()):
            if sample['hr_is_causal'] == False:
                golden_collection.at[i, 'action_class'] = ''
                golden_collection.at[i, 'ac_reasoning'] = ''
                continue
            
            try:
                chat_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages = [
                        {"role": "system", "content": prompt_action.format(PERGUNTA=sample["question"])},
                    ],
                )
            except Exception as e:
                print(e)
                golden_collection.at[i, 'action_class'] = ''
                golden_collection.at[i, 'ac_reasoning'] = ''
                continue

            response = chat_completion.choices[0].message.content
            reasoning = response.split("RACIOCÍNIO:")[-1].strip()
            classification = response.split("RACIOCÍNIO:")[0].split("CATEGORIA:")[-1].strip()
            
            golden_collection.at[i, 'action_class'] = classification
            golden_collection.at[i, 'ac_reasoning'] = reasoning
        
        golden_collection.to_excel("./data_gen/unified_ptbr_gpt_GC2_CoT.xlsx", index=False)
        
        for source in ['reddit', 'wildchat', 'sharegpt']:
            golden_collection[golden_collection['source'] == source].to_excel(f"./data_gen/unified_ptbr_gpt_GC_{name_ref[source]}2_CoT.xlsx", index=False)
        
    else:
        golden_collection = pd.read_excel("./data_gen/unified_ptbr_gpt_GC2_CoT.xlsx")
        golden_collection.fillna({
            'hr_is_causal': '',
            'hr_action_class': '',
            'hr_pearl_class': '',
            'action_class': '',
            'ac_reasoning': '',
            'pearl_class': '',
            'pc_reasoning': ''
            }, inplace=True)

    client.close()
    
    return golden_collection

def inference_pearl(golden_collection: pd.DataFrame):
    '''
    This function generates inference for Pearl classification.
    
    Parameters:
    : golden_collection - Golden Collection dataset, before inference.
    
    Return: Golden Collection, updated with Chain of Thought results for Pearl classification.
    '''
    
    api_key = os.environ['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_key)
    
    name_ref = {'reddit': 'RD', 'wildchat': 'WC', 'sharegpt': 'SG'}

    prompt_pearl = get_prompt_cot_pearl()
    
    if not os.path.exists("./data_gen/unified_ptbr_gpt_GC3_CoT.xlsx"):
        for i, sample in tqdm(golden_collection.iterrows()):
            if sample['hr_is_causal'] == False:
                golden_collection.at[i, 'pearl_class'] = ''
                golden_collection.at[i, 'pc_reasoning'] = ''
                continue
            
            try:
                chat_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages = [
                        {"role": "system", "content": prompt_pearl.format(PERGUNTA=sample["question"])},
                    ],
                )
            except Exception as e:
                print(e)
                golden_collection.at[i, 'pearl_class'] = ''
                golden_collection.at[i, 'pc_reasoning'] = ''
                continue

            response = chat_completion.choices[0].message.content
            reasoning = response.split("RACIOCÍNIO:")[-1].strip()
            classification = response.split("RACIOCÍNIO:")[0].split("CATEGORIA:")[-1].strip()
            
            golden_collection.at[i, 'pearl_class'] = classification
            golden_collection.at[i, 'pc_reasoning'] = reasoning
        
        golden_collection.to_excel("./data_gen/unified_ptbr_gpt_GC3_CoT.xlsx", index=False)
        
        for source in ['reddit', 'wildchat', 'sharegpt']:
            golden_collection[golden_collection['source'] == source].to_excel(f"./data_gen/unified_ptbr_gpt_GC_{name_ref[source]}3_CoT.xlsx", index=False)
        
    else:
        golden_collection = pd.read_excel("./data_gen/unified_ptbr_gpt_GC3_CoT.xlsx")
        golden_collection.fillna({
            'hr_is_causal': '',
            'hr_action_class': '',
            'hr_pearl_class': '',
            'action_class': '',
            'ac_reasoning': '',
            'pearl_class': '',
            'pc_reasoning': ''
            }, inplace=True)

    client.close()
    
    return golden_collection

def get_prompt_cot_causality():
    '''
    This function read the Chain of Thought prompt file for Causality.
    
    Parameters: -
    
    Return: Prompt for CoT Causality.
    '''
    
    with open('./prompts/cot_causality.txt', 'r') as file:
        prompt = file.read()
    
    return prompt

def get_prompt_cot_actions():
    '''
    This function read the Chain of Thought prompt file for Action Class.
    
    Parameters: -
    
    Return: Prompt for CoT Action Class.
    '''

    with open('./prompts/cot_actions.txt', 'r') as file:
        prompt = file.read()
    
    return prompt

def get_prompt_cot_pearl():
    '''
    This function read the Chain of Thought prompt file for Pearl Class.
    
    Parameters: -
    
    Return: Prompt for CoT Pearl Class.
    '''

    with open('./prompts/cot_pearl.txt', 'r') as file:
        prompt = file.read()
    
    return prompt