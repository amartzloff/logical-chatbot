#import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForCausalLM, pipeline


def get_answer(question, context):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased-distilled-squad")
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    model.eval()
    bert_question_answering = pipeline("question-answering", model=model, tokenizer=tokenizer)
    output = bert_question_answering({'question': question, 'context': context})
    return output['answer']


def get_chitchat(message, temperature):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")
    # encodes the input and add end of string token
    input_ids = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
    bot_input_ids = input_ids
    # uncomment this below in bot_input_ids to concatenate new user input with chat history
    # torch.cat([chat_history_ids, input_ids], dim=-1) if step > 0 else input_ids
    # generates a bot response
    chat_history_ids = model.generate(bot_input_ids,
                                      max_length=1000,
                                      do_sample=True,
                                      top_p=0.95,
                                      top_k=0,
                                      temperature=temperature,
                                      pad_token_id=tokenizer.eos_token_id
                                      )
    output = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    return output