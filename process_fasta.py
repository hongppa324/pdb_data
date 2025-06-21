import requests
import json
from tqdm import tqdm
import time

import numpy as np
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

# .env에서 API key 가져오기
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# 저장된 PDB ID 기록 파일 경로
PDB_LOG_FILE = "pdb_ids_svaed.json"

# pdb id 가져오는 함수 (n개씩)
def get_pdb_ids(n=100, start=0):
    # rcsb api url
    url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # 요청 query
    query = {
        "query": {
            "type": "terminal",
            "service": "text",
            "parameters": {
                "attribute": "entity_poly.rcsb_entity_polymer_type",
                "operator": "exact_match",
                "value": "Protein"
            }
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {
                "start": start,
                "rows": n
            }
        }
    }

    headers = {"Content-Type": "application/json"}

    response = requests.post(url, data=json.dumps(query), headers=headers)
    if response.status_code == 200:
        result = response.json()
        return [item["identifier"] for item in result.get("result_set", [])]
    else:
        print("PDB ID 요청 실패:", response.status_code)
        print(response.text)
        return []

# FASTA sequence 다운로드하는 함수
def download_fasta(pdb_id):
    url = f"https://www.rcsb.org/fasta/entry/{pdb_id}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.text
    return None

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document

# Document 생성하는 함수
def create_documents_from_fastas(pdb_ids, log_fn=None):
    documents = []
    for i, pdb_id in enumerate(pdb_ids):
        if log_fn:
            log_fn(f"📥 {pdb_id} FASTA 다운로드 중...")
        
        fasta = download_fasta(pdb_id)
        if fasta:
            lines = fasta.strip().splitlines()
            header = lines[0].replace(">", "").strip()
            sequence = ''.join(lines[1:])

            # header parsing
            # 예시: 101M_1|Chain A|MYOGLOBIN|Physeter catodon (9755)
            parts = header.split("|")
            if len(parts) == 4:
                sub_id = parts[0].strip()
                chain = parts[1].strip()
                protein = parts[2].strip()
                description = parts[3].strip()
            else:
                # 형식이 안 맞는 경우 기본값
                sub_id = header
                chain = protein = description = "Unknown"

            metadata = {
                "pdb_id": pdb_id,
                "sub_id": sub_id,
                "chain": chain,
                "protein": protein,
                "description": description,
                "sequence": sequence
            }

            doc = Document(page_content=sequence, metadata=metadata)
            documents.append(doc)

        time.sleep(0.1)
    return documents

# 저장된 pdb id 조회하는 함수
def load_saved_pdb_ids():
    if os.path.exists(PDB_LOG_FILE):
        with open(PDB_LOG_FILE, "r") as f:
            return set(json.load(f))
    return set()

# 저장된 pdf id 갱신하는 함수
def saved_pdb_ids(pdb_ids_set):
    with open(PDB_LOG_FILE, "w") as f:
        json.dump(sorted(list(pdb_ids_set)), f)

# FAISS 생성 및 merge하는 함수
def build_vectorstore(batch_size=100, total=1000, save_path="./vector_db/fasta", log_fn=print, progress_bar=None, should_continue=lambda: True):
    
    # embedding
    embeddings = OpenAIEmbeddings()

    # 저장된 pdb id 리스트 가져오기
    saved_ids = load_saved_pdb_ids()

    # 초기 vectorstore
    if os.path.exists(save_path):
        print("기존 벡터스토어 로드 중...")
        vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = None

    start_offset = len(saved_ids)
    total_batches = total // batch_size
    if progress_bar:
        progress_bar.progress(0.0)

    for i, start in enumerate(range(start_offset, start_offset + total, batch_size)):
        if not should_continue():
            log_fn("⏹️ 중단 요청 감지됨. 벡터스토어 생성을 종료합니다.")
            break        
        
        log_fn(f"\n🔄 {start} ~ {start + batch_size} PDB ID 처리 중...")
        pdb_ids = get_pdb_ids(n=batch_size, start=start)

        # 중복 제거
        new_ids = [pid for pid in pdb_ids if pid not in saved_ids]
        if not new_ids:
            log_fn("📭 수집할 새로운 PDB ID가 없습니다.")
            continue
        
        documents = create_documents_from_fastas(new_ids, log_fn=log_fn)
        if not documents:
            log_fn("⚠️ 문서 없음. 다음 배치로.")
            continue

        new_store = FAISS.from_documents(documents, embedding=embeddings)

        if vectorstore is None:
            vectorstore = new_store
        else:
            vectorstore.merge_from(new_store)

        # 매 배치마다 저장
        saved_ids.update(new_ids)
        saved_pdb_ids(saved_ids)

        vectorstore.save_local(save_path)
        log_fn(f"✅ 저장 완료: {len(documents)}개 문서, 총 저장된 ID 수: {len(saved_ids)}")

        if progress_bar:
            progress_bar.progress((i + 1) / total_batches)

    log_fn("🏁 벡터스토어 생성 완료!")

# 실행
if __name__ == "__main__":
    build_vectorstore(batch_size=100, total=1000)

# 저장된 vectorstore 확인하는 함수
def inspect_vectorstore(save_path="./vector_db/fasta"):
    embeddings = OpenAIEmbeddings()
    if os.path.exists(save_path):
        store = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)

        # 문서 수 확인
        num_docs = len(store.docstore._dict)  # 내부 dictionary 사용
        print(f"🔍 저장된 벡터스토어 문서 수: {num_docs}")

        # 일부 메타데이터 출력
        print("📌 샘플 메타데이터:")
        for i, doc in enumerate(store.docstore._dict.values()):
            print(doc.metadata)
            if i >= 4:
                break
    else:
        print("❌ 벡터스토어 경로가 존재하지 않습니다.")

# 확인하는 함수 실행
if __name__ == "__main__":
    inspect_vectorstore()