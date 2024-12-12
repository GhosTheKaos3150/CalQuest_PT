import os
import openai
import pandas as pd

def generate_inference(dataset: pd.DataFrame):
    '''
    This function uses GPT to infer the 5W-2H classifications that are missing and were not classified by
    using regex methods on previous steps.
    
    Parameters:
    : dataset - Full CalQuest.PT dataset, with all data sources.
    
    Return: Full CaulQuest.PT dataset, with '5w2h' collumn completed.
    '''
    
    api_key = os.environ['OPENAI_API_KEY']
    client = openai.OpenAI(api_key=api_key)
    
    PROMPT_INFERENCE = '''A pergunta que se segue foi feita por um humano e você deve classificar esta pergunta em uma\
    das categorias a seguir:

    O quê: São perguntas que buscam uma ação a ser executada (qual a ação a ser executada) ou problema a ser solucionado. \
    Também pode se referir a identificação de algo ou de um objeto.
    Por quê: Perguntas que buscam justificativas de algo, qual a motivação ou objetivo de algo ser feito ou ter acontecido.
    Quem: Pergunta sobre quem executou a ação, quem é o responsável ou culpado.
    Onde: Pergunta onde a ação foi ou vai ser executada ou onde algo aconteceu ou vai acontecer. Também pode se referir a \
    pedidos de indicação sobre lugares com fins específicos, como onde encontrar emprego ou melhores locais para morar.
    Quando: Pergunta quando uma ação foi ou será executada, perguntas de ordem cronológica em geral.
    Como: Pergunta como algo aconteceu ou vai acontecer, como uma ação foi ou será executada. Também pode se referir a \
    saber o estado de algo ou métodos a serem empregados para alcançar um fim.
    Quanto: São perguntas que envolvem saber o custo de algo, podendo ser em relação a custo financeiro, de tempo, de recursos, etc.

    Exemplos:
    Pergunta: Qual a coisa mais absurda que vocês já viram alguém defender nos comentários do Instagram? Categoria: O quê
    Pergunta: Por que a morte de Elio de Angelis foi um grande baque para a Brabham, que começou a entrar em declínio até fechar as portas? Categoria: Por quê
    Pergunta: Quem usa o Docker concorda? Categoria: Quem
    Pergunta: Onde encontrar congressos e eventos para ir? Categoria: Onde
    Pergunta: Quando você se deu conta que seu nível de senioridade varia de empresa para empresa? Categoria: Quando
    Pergunta: Como está o mercado financeiro e carreira de economia fora do eixo SP-RJ? Categoria: Como
    Pergunta: Por quanto tempo você dorme? Categoria: Quanto
    Pergunta: Qual banco de dados vocês recomendam estudar? Categoria: O quê
    Pergunta: Por que macenaria é uma área que rende muito dinheiro? Categoria: Por quê
    Pergunta: De quem vocês mais sentem falta? Categoria: Quem
    Pergunta: Alguém teria dicas de onde encontrar uma vaga de estagio na área de figurino? Categoria: Onde
    Pergunta: Quando foi inaugurada a primeira universidade nos EUA? E no Brasil? Categoria: Quando
    Pergunta: Emprego remoto em portugal, como receber sendo CNPJ? Categoria: Como
    Pergunta: Quanto tempo demora pra enviar os equipamentos? Categoria: Quanto

    Segue a pergunta: {PERGUNTA} e solicito que você classifique em uma das sete categorias acima detalhadas: O quê, Por quê, Quem, Onde, \
    Quando, Como, Quanto. Caso você não consiga classificar em uma dessas categorias, classifique como "Outro". Não responda a pergunta apresentada, \
    retorne apenas a categoria no seguinte formato:
    CATEGORIA:'''

    if not os.path.exists("./data_gen/class_5w2h.xlsx"):
        for i, sample in dataset.iterrows():
            if dataset[dataset['5w2h'] != '']:
                continue
            
            try:
                chat_completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages = [
                        {"role": "system", "content": PROMPT_INFERENCE.format(PERGUNTA=sample["question"])},
                    ],
                )
            except Exception as e:
                print(e)
                dataset.at[i, '5w2h'] = "Outro"
                dataset.at[i, '5w2h_reasoning'] = ""
                continue

            response = chat_completion.choices[0].message.content
            reasoning = response.split("RACIOCÍNIO:")[-1].strip()
            classification = response.split("RACIOCÍNIO:")[0].split("CATEGORIA:")[-1].strip()
            
            dataset.at[i, '5w2h'] = classification
            dataset.at[i, '5w2h_reasoning'] = reasoning
        
        dataset.to_excel("./data_colins_2025/Experimento Amostra 3/class_5w2h.xlsx", index=False)
        
    else:
        dataset = pd.read_excel("./data_colins_2025/Experimento Amostra 3/class_5w2h.xlsx")
        dataset.fillna({'5w2h': ''}, inplace=True)
        
    client.close()
    
    return dataset