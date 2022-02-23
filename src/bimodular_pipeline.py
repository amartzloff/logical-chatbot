from analyse import *
from generation import *

message = """Good muffins cost $3.88 in New York.  Please buy me... two of them. Thanks. I love you so much. Where have you been ? Oh my god this is so good ! How are you doing today?"""
#"""I was with my grand ma yesterday morning. She is so cute. She drove me to school. Her new car is so beautiful ! Who owns the car?"""
#"""How are you feeling?"""
#"""What is the name of Barack Obama's wife?"""
#"""Good muffins cost $3.88 in New York.  Please buy me... two of them. Thanks. I love you so much. Where have you been ? Oh my god this is so good ! How are you doing today?"""

# gets the sentence on which we want to make the prediction
last_sentence, context = get_input(message)
print(">> User :    " + context)
print("             " + last_sentence)

# gets prediction for the entered sentence
sum_up, binary_label, text_label = classif(last_sentence, 0)

if binary_label == 0:
    chitchat_answer = get_chitchat(message, 0.5)
    print(">> Machine : " + chitchat_answer)
elif binary_label == 1:
    qa_answer = get_answer(last_sentence, context)
    print(qa_answer)
