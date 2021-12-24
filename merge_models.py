from models.chatbot import ChatBot
from models.mrc import mrc_prompt
from models.retrieval.retrieval import *

# 모델 로딩
'''
retrieval class
mrc class
chatbot class
'''
retriever_argument = parse_argument()
retriever = SparseRetrieval(
        tokenize_fn=tokenize,
        data_path="data/retrieval/klue-mrc-v1.1",
        context_path="contexts.json",
        args=retriever_argument)

# argument parsing

# 질문 서버에서 받아옴
document_query = "" #문서 검색
chat_query = "BMW 대표는?"  #서버에서 받아와야함
selected_document = """BMW 코리아(대표 한상윤)는 창립 25주년을 기념하는 ‘BMW 코리아 25주년 에디션’을 한정 출시한다고 밝혔다. 이번 BMW 코리아 25
주년 에디션(이하 25주년 에디션)은 BMW 3시리즈와 5시리즈, 7시리즈, 8시리즈 총 4종, 6개 모델로 출시되며, BMW 클래식 모델들로 선보인 바 있는 헤리티지 컬러
가 차체에 적용돼 레트로한 느낌과 신구의 조화가 어우러진 차별화된 매력을 자랑한다. 먼저 뉴 320i 및 뉴 320d 25주년 에디션은 트림에 따라 옥스포드 그린(50대
 한정) 또는 마카오 블루(50대 한정) 컬러가 적용된다. 럭셔리 라인에 적용되는 옥스포드 그린은 지난 1999년 3세대 3시리즈를 통해 처음 선보인 색상으로 짙은 녹
색과 풍부한 펄이 오묘한 조화를 이루는 것이 특징이다. M 스포츠 패키지 트림에 적용되는 마카오 블루는 1988년 2세대 3시리즈를 통해 처음 선보인 바 있으며, 보
랏빛 감도는 컬러감이 매력이다. 뉴 520d 25주년 에디션(25대 한정)은 프로즌 브릴리언트 화이트 컬러로 출시된다. BMW가 2011년에 처음 선보인 프로즌 브릴리언트
 화이트는 한층 더 환하고 깊은 색감을 자랑하며, 특히 표면을 무광으로 마감해 특별함을 더했다. 뉴 530i 25주년 에디션(25대 한정)은 뉴 3시리즈 25주년 에디션
에도 적용된 마카오 블루 컬러가 조합된다. 뉴 740Li 25주년 에디션(7대 한정)에는 말라카이트 그린 다크 색상이 적용된다. 잔잔하면서도 오묘한 깊은 녹색을 발산
하는 말라카이트 그린 다크는 장식재로 활용되는 광물 말라카이트에서 유래됐다. 뉴 840i xDrive 그란쿠페 25주년 에디션(8대 한정)은 인도양의 맑고 투명한 에메
랄드 빛을 연상케 하는 몰디브 블루 컬러로 출시된다. 특히 몰디브 블루는 지난 1993년 1세대 8시리즈에 처음으로 적용되었던 만큼 이를 오마주하는 의미를 담고 
있다."""


def query2passages(query):
    # retrieval 객체가 query를 받아 args.topk에 해당하는 passages return
    scores, indices = retriever.retrieve(query, topk=retriever_argument.topk)
    return scores, indices


# MRC 모델에 query 넣어준 후 answer에 결과 저장
def query2answer(query, passage):

    # MRC 모델에 query, passage 넣고 나오는 결과 리턴
    answer = mrc_prompt.main(query, passage)
    # 만약에 MRC에서 no_answer(cls token)이 뽑히면 "NoAnswer"리턴

    return answer

def query2chatbot(query):
    answer = ChatBot.getAnswer(query)
    print("나의 질문 : ", query)
    print("나의 대답 : ", answer)
    return answer

def conversation(query, passage):
    print('query', query)
    print('passage', passage)
    if len(passage.strip()) == 0: # 문서 선택 안됨
        answer = query2chatbot(query) # 이 변수 클라이언트에 보내셈
    else: # 문서 선택 됨
        answer = query2answer(query, passage) #MRC 모델 돌리기
        if answer == "no_answer": #답 없으면 챗봇 돌리기
            answer = query2chatbot(query)
    
    return answer
            

if __name__ == "__main__":
    # 1. 문서 검색한 걸 토대로 문서 다섯개 가져오기 - 근영 구현
    document_query = input('질문을 입력하세요 : ')
    _, passages = query2passages(document_query)
    # 2. 선택된 문서에서 질문 받아오기
    selected_document = passages[int(input('번호를 입력하세요 : '))]

    # 3. noanswer 나오면 챗봇 돌리기
    while(1):
        chat_query = input('대화를 입력하세요 : ')
        if len(selected_document) == 0: # 문서 선택 안됨
            answer = query2chatbot(chat_query) # 이 변수 클라이언트에 보내셈
        else: # 문서 선택 됨
            answer = query2answer(chat_query, selected_document) #MRC 모델 돌리기
            if answer == "no_answer": #답 없으면 챗봇 돌리기
                answer = query2chatbot(chat_query)

    # 4. answer 클라이언트로 보내시오