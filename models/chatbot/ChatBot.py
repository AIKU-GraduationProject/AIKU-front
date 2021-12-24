import numpy as np
import torch
#from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity

# #모델 로드
# MODEL_NAME = "bert-base-multilingual-cased"
# tokenizer = torch.load("data/chatbot/tokenizer")
# model = torch.load("models/chatbot/chat_bot_bert_model")
# dataset_cls_hidden = np.load('models/chatbot/ChatBot_numpy_model.npy')
# chatbot_Answer = np.load('models/chatbot/Answer_data.npy')


#모델 로드
MODEL_NAME = "bert-base-multilingual-cased"
tokenizer = torch.load("/home/konkuk/Desktop/merge/data/chatbot/tokenizer")
model = torch.load("/home/konkuk/Desktop/merge/models/chatbot/chat_bot_bert_model")
dataset_cls_hidden = np.load('/home/konkuk/Desktop/merge/models/chatbot/ChatBot_numpy_model.npy')
chatbot_Answer = np.load('/home/konkuk/Desktop/merge/models/chatbot/Answer_data.npy')

print("Load compltete")

def get_cls_token(sent_A):
    model.eval()
    tokenized_sent = tokenizer(
            sent_A,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=True,
            max_length=128
    )
    with torch.no_grad():# 그라디엔트 계산 비활성화
        outputs = model(    # **tokenized_sent
            input_ids=tokenized_sent['input_ids'],
            attention_mask=tokenized_sent['attention_mask'],
            token_type_ids=tokenized_sent['token_type_ids']
            )
    logits = outputs.last_hidden_state[:,0,:].detach().cpu().numpy()
    return logits

def getAnswer(query):
    query_cls_hidden = get_cls_token(query)
    cos_sim = cosine_similarity(query_cls_hidden, dataset_cls_hidden)
    top_question = np.argmax(cos_sim)
    answer = chatbot_Answer[top_question]
    return answer

# query = ""
# print('나의 질문: ', query)
# print('저장된 답변: ', getAnswer(query))

def get_scores():
    top1 = 0
    top3 = 0
    top5 = 0
    print(f"Total dataset(cls) size = {len(dataset_cls_hidden)}")
    print(f"Total dataset(answer) size = {len(chatbot_Answer)}")
    print(dataset_cls_hidden.shape)
    print(chatbot_Answer.shape)
    for idx in tqdm(range(len(dataset_cls_hidden)), desc='Evaluation', total=len(dataset_cls_hidden)):
        hidden = dataset_cls_hidden[idx, :].reshape(1, -1)
        cos_sim = cosine_similarity(hidden, dataset_cls_hidden)
        cos_sim = cos_sim.squeeze()
        topk_idx = torch.topk(torch.tensor(cos_sim), 5).indices
        if idx in topk_idx[0]:
            top1 += 1
            top3 += 1
            top5 += 1
            continue
        
        if idx in topk_idx[:3]:
            top3 += 1
            top5 += 1
            continue
        
        if idx in topk_idx:
            top5 += 1
            continue
    
    print(f"Top 1 : {top1} / {len(chatbot_Answer)} = {top1/len(chatbot_Answer)}")
    print(f"Top 3 : {top3} / {len(chatbot_Answer)} = {top3/len(chatbot_Answer)}")
    print(f"Top 5 : {top5} / {len(chatbot_Answer)} = {top5/len(chatbot_Answer)}")
    
get_scores()