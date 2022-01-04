import json
import pandas as pd

def get_qa_df(filepath, label):
  # loads json file
  with open(filepath) as f:
    data = json.load(f)
  # extracts questions
  data_dicts = data['data']
  # creates list of questions
  inputs = []
  number__dicts = len(data['data'])
  for i in range(number__dicts):
    paragraphs = data_dicts[i]['paragraphs']
    number_paragraph = len(paragraphs)
    for j in range(number_paragraph):
      qas = paragraphs[j]['qas']
      number_quas = len(qas)
      for k in range(number_quas):
        inputs.append(qas[k]['question'])
  qa_number = len(inputs)
  labels = [label for i in range(qa_number)]
  frame = {'Input': inputs, 'Label': labels}
  qa_df = pd.DataFrame(frame)
  return qa_df

def get_chitchat_df(filepath, label):
  # loads json file
  with open(filepath) as f:
    data = json.load(f)
  # extracts data
  qna_list = data['qnaList']
  # creates list of chitchats
  inputs = []
  qna_list_length = len(qna_list)
  for i in range(qna_list_length) :
    question = qna_list[i]['questions']
    inputs += question
  chitchat_number = len(inputs)
  labels = [label for i in range(chitchat_number)]
  frame = {'Input': inputs, 'Label': labels}
  chitchat_df = pd.DataFrame(frame)
  return chitchat_df

def create_dataset(df1,df2):
  # ne retourne rien car juste cela créer un fichier csv !
  return none