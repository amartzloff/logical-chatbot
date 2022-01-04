from data_loading import *

# creates the dataframes
chitchat_df = get_chitchat_df('../data/chitchat.json', 0)
qa_df = get_qa_df('../data/qa.json', 1)

# joins dataframes to csv file : our dataset
create_dataset(chitchat_df, qa_df, '../out', 'dataset.csv')
