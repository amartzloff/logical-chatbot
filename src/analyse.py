import torch
import sklearn
import numpy as np
from sklearn.metrics import matthews_corrcoef
from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

# checks torch and transformers versions

#print("Version:",torch.__version__)
#import transformers
#print("Version:",transformers.__version__)
#print("Version:",sklearn.__version__)

# checks if a GPU is available

#print("Has GPU:",torch.cuda.is_available())

# checks if torch works

#print("Random tensor:",torch.rand(10,device="cpu"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
analysis_model = '../analysis_model'

# Load our trained model and vocabulary we have fine-tuned
model = BertForSequenceClassification.from_pretrained(analysis_model, use_auth_token=True)
tokenizer = BertTokenizer.from_pretrained(analysis_model)

# Copy the model to the GPU
model.to(device)

# function that gives the equivalence integer/text for a label
def my_label(label):
  if label == 0:
    return("Chitchat")
  elif label == 1:
    return("Q&A")

# function that gives the text and integer predicted label for a given sentence
def classif(phrase1, label1, phrase2="How are you", label2=0):
    test_raw = [phrase1, phrase2]
    test_labels = [label1, label2]
    sentences = test_raw
    labels = test_labels
    input_ids = []
    attention_masks = []
    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
            sent,  # Sentence to encode.
            add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
            max_length=64,  # Pad & truncate all sentences.
            padding='max_length',
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
        )
        # adds encoded sentence to list
        input_ids.append(encoded_dict['input_ids'])
        # adds attention mask (differentiates padding from non-padding) to list
        attention_masks.append(encoded_dict['attention_mask'])
    # converts lists into tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    # sets the batch size.
    batch_size = 32
    # creates dataLoader
    prediction_data = TensorDataset(input_ids, attention_masks, labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)
    # puts model in evaluation mode
    model.eval()
    # tracks variables
    predictions, true_labels = [], []
    # predicts
    for batch in prediction_dataloader:
        # adds batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # unpacks inputs from dataloader
        b_input_ids, b_input_mask, b_labels = batch
        # tells the model not to compute gradients
        with torch.no_grad():
            # does forward pass (calculates logit predictions)
            result = model(b_input_ids,
                           token_type_ids=None,
                           attention_mask=b_input_mask,
                           return_dict=True)
        logits = result.logits
        # moves logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        # store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)
    matthews_set = []
    # evaluates each test batch using Matthew's correlation coefficient
    new_pred_labels = []
    for i in range(len(true_labels)):
        pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
        new_pred_labels.append(pred_labels_i)
        # calculates coefficient for batch
        matthews = matthews_corrcoef(true_labels[i], pred_labels_i)
        matthews_set.append(matthews)
    # print("The sentence was : " + sentences[0] + "\n" + "True label : " + my_label(true_labels[0][0]) + "\n" + "Predicted label : " + my_label(new_pred_labels[0][0]))
    to_print = """ The sentence was "{}" and was labeled {} i.e. "{}"  \n It got predicted as {} i.e. "{}" """.format(
        phrase1, label1, my_label(label1), new_pred_labels[0][0], my_label(new_pred_labels[0][0]))
    return to_print, new_pred_labels[0][0], my_label(new_pred_labels[0][0])