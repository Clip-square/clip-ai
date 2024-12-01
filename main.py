from sentence_transformers import SentenceTransformer, util
import numpy as np
import openai
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import os

# 환경변수 로드
load_dotenv(dotenv_path="/Users/dgsw8th66/Desktop/전공 공부 폴더/Clip/ChatGPT.env")  # .env 파일 경로 지정

# API 키 확인
openai.api_key = os.getenv("OPENAI_API_KEY")

# FastAPI 애플리케이션 생성
app = FastAPI()

# SentenceTransformer 모델 로드
model = SentenceTransformer('all-MiniLM-L6-v2')

# 벡터 DB 로드
minutes_vector_db = np.load("/Users/dgsw8th66/Desktop/전공 공부 폴더/Clip/minutes_vector_db.npz", allow_pickle=True)
summary_vector_db = np.load("/Users/dgsw8th66/Desktop/전공 공부 폴더/Clip/summary_vector_db.npz", allow_pickle=True)

# 요청 데이터 모델 정의
class MeetingRequest(BaseModel):
    topic: str
    subTopicNamelist: list
    speechList: list
    date: str

# 벡터 DB에서 관련 요약 스타일 데이터 검색
def retrieve_summary_style(text, vector_db):
    embeddings = vector_db['data']
    metadata = vector_db['indices']
    
    # 입력 텍스트의 벡터 추출
    text_embedding = model.encode(text)  # (1, 384)
    
    # embeddings가 2D 배열인지 확인 후 (N, 384) 형태로 처리
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 384)  # (N, 384)로 변환
    
    # 데이터 타입을 맞추기 (float32로 변환)
    text_embedding = text_embedding.astype(np.float32)
    embeddings = embeddings.astype(np.float32)
    
    # 유사도 계산
    similarity = util.cos_sim(text_embedding, embeddings)  # (1, N) 형태의 결과
    
    # 가장 높은 유사도를 가진 인덱스 찾기
    best_match_idx = similarity.argmax()
    
    # 관련된 요약 스타일 반환
    return metadata[best_match_idx]

# 벡터 DB에서 회의록 주제에 대한 정보를 검색하는 함수 정의
def retrieve_minutes_topic(text, vector_db):
    embeddings = vector_db['data']
    metadata = vector_db['indices']
    
    # 입력 텍스트의 벡터 추출
    text_embedding = model.encode(text)  # (1, 384)
    
    # embeddings가 2D 배열인지 확인 후 (N, 384) 형태로 처리
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(-1, 384)  # (N, 384)로 변환
    
    # 유사도 계산
    similarity = util.cos_sim(text_embedding, embeddings)  # (1, N) 형태의 결과
    
    # 가장 높은 유사도를 가진 인덱스 찾기
    best_match_idx = similarity.argmax()
    
    # 관련된 회의록 주제 반환
    return metadata[best_match_idx]

# GPT-4 API 호출 (비동기 처리)
async def generate_summary(prompt):
    response = await openai.ChatCompletion.acreate(
        model="gpt-4",  # GPT-4 모델 사용
        messages=[
            {"role": "system", "content": "당신은 전문 회의록 작성자입니다."},
            {"role": "user", "content": prompt}
        ]
    )
    return response['choices'][0]['message']['content']

# 회의록 생성 함수
async def create_minutes(topic, subTopicNamelist, speechList, date):
    # 소주제별로 요약 처리
    summaries = []
    for sub_topic, speech in zip(subTopicNamelist, speechList):
        # 벡터 DB에서 요약 스타일 검색
        summary_style = retrieve_summary_style(speech, summary_vector_db)
        
        # GPT-4 프롬포트 준비
        prompt = f"""
        다음 스타일로 회의 내용을 요약하세요:
        {summary_style}
        
        소주제: {sub_topic}
        내용: {speech}
        """
        
        # GPT-4로 요약 생성
        summary = await generate_summary(prompt)
        summaries.append(f"## {sub_topic}\n{summary}")

    # `minutes_vector_db`를 활용하여 회의록 주제에 대한 정보를 얻어오기
    minutes_topic_info = retrieve_minutes_topic(topic, minutes_vector_db)
    
    # 종합 정리 부분을 적절하게 가공 (주제에 맞는 텍스트로 변환)
    minutes_topic_info_text = "\n".join(minutes_topic_info) if isinstance(minutes_topic_info, list) else minutes_topic_info

    # 최종 회의록 마크다운 형식으로 작성
    minutes = f"# {topic}\n\n**회의 일시**: {date}\n\n" + "\n\n".join(summaries) + f"\n\n---\n\n### 종합 정리\n{minutes_topic_info_text}"
    return {"minutes": minutes}

# FastAPI 엔드포인트 정의
@app.post("/generate_minutes/")
async def generate_minutes_endpoint(meeting: MeetingRequest):
    minutes = await create_minutes(
        meeting.topic,
        meeting.subTopicNamelist,
        meeting.speechList,
        meeting.date
    )
    return minutes
