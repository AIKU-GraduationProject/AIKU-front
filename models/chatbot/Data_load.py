# import torch
# from transformers import AutoModel, AutoTokenizer

# MODEL_NAME = "bert-base-multilingual-cased"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModel.from_pretrained(MODEL_NAME)

# print("Model")

import csv
import numpy as np

f = open('./ChatBot_DataSet/ChatbotData.csv', 'r', encoding='utf-8')
rdr = csv.reader(f)
cq = []
ca = []
for idx,i in enumerate(rdr):
    if idx == 0: continue
    cq.append(i[0])
    ca.append(i[1])
f.close()

import pandas as pd

df = pd.read_excel('./ChatBot_DataSet/data.xlsx')
cq_1 = []
ca_1 = []

df = df.fillna(0)

cnt = 0
for i in range(len(df['챗봇'])):
    if df['챗봇'][i] == 0:
        continue
    cq_1.append(df['유저'][i])
    ca_1.append(df['챗봇'][i])


file = open("./ChatBot_DataSet/KETI_Data.txt", 'r')
strings = file.readlines()
questions = []
answers = []

for i in range(0,len(strings), 2):
    questions.append(strings[i].split('\t')[1][:-1])
    answers.append(strings[i+1].split('\t')[1][:-1])

a =["감기 걸린 거 같아", "감기 걸렸어", "감기 걸린 것 같아", "너는 뭐해", "뭐해", "너는 뭘 잘하니?", "너는 뭘 잘하니", "배 안고파", "배 고프지 않니?", "민덕기", "너무 우울해", "우울할땐 어떻게 해?", "지칠땐 어떻게 해?", "졸릴땐 어떻게 해?", "취준 힘들다.", "힘들다", "수고했어", "수고했어.", "난 언제 쉴 수 있을까","난 언제 쉴 수 있을까?", "난 언제 잘 수 있을까?", "난 언제 잘 수 있을까", "난 언제 죽을까?", "난 언제 죽을까", "죽여줘", "죽여줘?", "오늘 바빠", "밥줘", "귀찮아", "자기야", "자기야?"]
q =["이럴 때 쉬는 게 중요해요.", "이런, 푹 쉬어야 얼른 나아요.", "푹 쉬어야 해요.", "저는 문서 읽고 대답해줘요.", "채팅중이잖아요.", "저는 대답을 잘해요.", "저는 못하는게 없어요.", "저는 항상 배고파요.", "저는 항상 배고파요.", "교수님 짱짱", "잘하고 있어요. 당당해지세요.","기분 전환을 해보아요!", "잠시 쉬어가는 것도 답이 될 수 있어요.", "커피 한잔 어때요?", "조금만 더 버텨보세요.", "원래 쉬운게 없어요.", "별거 아니에요.", "별거 아니에요.", "바쁘게 사는게 멋있어요.", "바쁘게 사는게 멋있어요.", "하루 6시간은 자야해요.", "하루 6시간은 자야해요.", "사람에게 잊혀졌을때요.", "사람에게 잊혀졌을때요.", "싫어요.", "죄송합니다.", "바쁘면 좋은거죠.", "지금 당장 거울 앞에 서 보세요.", "버릴 건 버리세요.", "저는 자기가 아닙니다.", "자기 아닙니다." ]
a2 = ["사과를 먹고싶어", "사과 먹고싶어", "알겠어", "나 힘들어", "졸려", "축구 하고싶어", "농구 하고싶어", "산책 하고싶어", "산책하러 갈까?", "야식 먹고싶어", "야식 먹을래", "점심 메뉴 추천해줘", "점심 뭐 먹지?", "점심 뭐 먹을까?", "저녁 뭐 먹지?", "저녁 뭐 먹을까?", "저녁 메뉴 추천해줘"]
q2 = ["저도 사과 좋아해요.", "과일 가게에 가서 사드세요.", "네~", "힘내세요.", "피곤하면 주무세요.", "저랑 같이해요!", "저랑 같이해요!", "지금 당장 하고오세요.", "좋아요!", "야식은 몸에 해로워요", "야식은 몸에 해로워요", "뜨끈한 국밥 어때요?", "뜨끈한 국밥 어때요?", "뜨끈한 국밥 어때요?", "오늘 하루 수고한 당신에게 고기를 사줘봐요.", "오늘 하루 수고한 당신에게 고기를 사줘봐요.", "오늘 하루 수고한 당신에게 고기를 사줘봐요."]
chatbot_Question = cq + cq_1+ a2 + a + questions + ['이름이 뭐야?','기차 타고 여행 가고 싶어','꿈이 이루어질까?','내년에는 더 행복해질려고 이렇게 힘든가봅니다', '간만에 휴식 중', '오늘도 힘차게!'] # 질문
chatbot_Answer =  ca + ca_1 + q2 + q + answers + ['정제윈입니다.','꿈꾸던 여행이네요.','현실을 꿈처럼 만들어봐요.','더 행복해질 거예요.', '휴식도 필요하죠', '아자아자 화이팅!!'] # 답변

print("Complete Data Input")

np.save("./Answer_data", chatbot_Answer)