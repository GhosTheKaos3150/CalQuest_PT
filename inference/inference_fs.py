import os
import openai
import pandas as pd
from tqdm import tqdm

def generate_inference(golden_collection: pd.DataFrame):
    '''
    TODO Documentação
    '''
    
    gc_infered_few_shot = inference_causality(golden_collection)
    gc_infered_few_shot = inference_actions(gc_infered_few_shot)
    gc_infered_few_shot = inference_pearl(gc_infered_few_shot)
    
    return gc_infered_few_shot

def inference_causality(golden_collection: pd.DataFrame):
    '''
    TODO Documentação
    '''
    
    api_key = os.environ['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_key)
    
    name_ref = {'reddit': 'RD', 'wildchat': 'WC', 'sharegpt': 'SG'}
    
    prompt_causal = get_prompt_fs_causality()
    
    if not os.path.exists("./data_gen/unified_ptbr_gpt_GC.xlsx"):
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
            
            golden_collection.at[i, 'is_causal'] = True if classification == "Causal" else False
            golden_collection.at[i, 'causal_reasoning'] = reasoning
        
        golden_collection.to_excel("./data_gen/unified_ptbr_gpt_GC.xlsx", index=False)
        
        for source in ['reddit', 'wildchat', 'sharegpt']:
            golden_collection[golden_collection['source'] == source].to_excel(f"./data_gen/unified_ptbr_gpt_GC_{name_ref[source]}.xlsx", index=False)
    
    else:
        golden_collection = pd.read_excel("./data_gen/unified_ptbr_gpt_GC.xlsx")
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
    TODO Documentação
    '''
    
    api_key = os.environ['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_key)
    
    name_ref = {'reddit': 'RD', 'wildchat': 'WC', 'sharegpt': 'SG'}
    
    prompt_action = get_prompt_fs_actions()

    if not os.path.exists("./data_gen/unified_ptbr_gpt_GC2.xlsx"):
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
        
        golden_collection.to_excel("./data_gen/unified_ptbr_gpt_GC2.xlsx", index=False)
        
        for source in ['reddit', 'wildchat', 'sharegpt']:
            golden_collection[golden_collection['source'] == source].to_excel(f"data_gen/unified_ptbr_gpt_GC_{name_ref[source]}2.xlsx", index=False)
        
    else:
        golden_collection = pd.read_excel("./data_gen/unified_ptbr_gpt_GC2.xlsx")
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
    TODO Documentação
    '''

    api_key = os.environ['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_key)
    
    name_ref = {'reddit': 'RD', 'wildchat': 'WC', 'sharegpt': 'SG'}
    
    prompt_pearl = get_prompt_fs_pearl()
    
    if not os.path.exists("./data_gen/unified_ptbr_gpt_GC3.xlsx"):
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
        
        golden_collection.to_excel("./data_gen/unified_ptbr_gpt_GC3.xlsx", index=False)
        
        for source in ['reddit', 'wildchat', 'sharegpt']:
            golden_collection[golden_collection['source'] == source].to_excel(f"data_gen/unified_ptbr_gpt_GC_{name_ref[source]}3.xlsx", index=False)
        
    else:
        golden_collection = pd.read_excel("./data_gen/unified_ptbr_gpt_GC3.xlsx")
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

def get_prompt_fs_causality():
    '''
    TODO Documentação
    '''

    with open('./prompts/few-shot_causality.txt', 'r') as file:
        prompt = file.read()
    
    return prompt

def get_prompt_fs_actions():
    '''
    TODO Documentação
    '''
    
    with open('./prompts/few-shot_actions.txt', 'r') as file:
        prompt = file.read()
    
    return prompt

def get_prompt_fs_pearl():
    '''
    TODO Documentação
    '''
    
    with open('./prompts/few-shot_pearl.txt', 'r') as file:
        prompt = file.read()
    
    return prompt