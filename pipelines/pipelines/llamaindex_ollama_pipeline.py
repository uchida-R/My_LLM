"""
title: Llama Index Ollama Pipeline
author: open-webui
date: 2024-05-30
version: 1.0
license: MIT
description: A pipeline for retrieving relevant information from a knowledge base using the Llama Index library with Ollama embeddings.
requirements: llama-index, llama-index-llms-ollama, llama-index-embeddings-ollama
"""
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode, TransformComponent
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

from typing import List, Union, Generator, Iterator
from schemas import OpenAIChatMessage
import os

from pydantic import BaseModel

class ChromaDB_LlamaIndexEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, ef: BaseEmbedding):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        return [node.embedding for node in
                self.ef([TextNode(text=doc) for doc in input])]
    
class Pipeline:

    class Valves(BaseModel):
        OLLAMA_HOST: str
        LLM_MODEL: str
        EMBED_MODEL: str
        VECDB_HOSTNAME: str
        VECDB_PORT: int

    def __init__(self):
        self.documents = None
        self.index = None

        # うまく環境変数を読み込めていない 実質ハードコーティング
        self.valves = self.Valves(
            **{
                "OLLAMA_HOST": os.getenv("OLLAMA_HOST", "http://ollama:11434"),
                "LLM_MODEL": os.getenv("LLM_MODEL", "gemma3:12b"),
                "EMBED_MODEL": os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-large"),
                "VECDB_HOSTNAME":os.getenv("VECDB_HOSTNAME","chromadb"),
                "VECDB_PORT": int(os.getenv("VECDB_PORT","8000"))
            }
        )

    async def on_startup(self):
        from llama_index.llms.ollama import Ollama
        from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader

        from llama_index.embeddings.huggingface import HuggingFaceEmbedding




        # 埋め込みモデルをダウンロードしている ollamaに統合すべきか検討
        Settings.embed_model = HuggingFaceEmbedding(model_name=self.valves.EMBED_MODEL,embed_batch_size=8,max_length=512)
        Settings.llm = Ollama(
            model=self.valves.LLM_MODEL,
            request_timeout=1000.0,
            context_window=8192,
            base_url=self.valves.OLLAMA_HOST,
        )
        # self.chroma_client = chromadb.HttpClient(host=self.valves.VECDB_HOSTNAME, port=int(self.valves.VECDB_PORT))
        self.chroma_client = chromadb.HttpClient(host="chromadb", port=int("8000")) #ハードコーティング 訂正予定

        


        # # This function is called when the server is started.
        # global documents, index

        # self.documents = SimpleDirectoryReader("/app/backend/data").load_data()
        # self.index = VectorStoreIndex.from_documents(self.documents)
        # pass

    async def on_shutdown(self):
        # This function is called when the server is stopped.
        pass

    def pipe(
        self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        # This is where you can add your custom RAG pipeline.
        # Typically, you would retrieve relevant information from your knowledge base and synthesize it to generate a response.
        
        from llama_index.core import StorageContext
        from llama_index.vector_stores.chroma import ChromaVectorStore
        from llama_index.core.base.llms.types import ChatMessage, MessageRole
        from llama_index.core.prompts import ChatPromptTemplate
        from llama_index.core import Settings,VectorStoreIndex


        # この部分はstartup時に一回やれば良さそう
        collection = self.chroma_client.get_or_create_collection(
            name="labs_rag", # コレクション名 コレクション名ごとにデータを分けることができる
            embedding_function=ChromaDB_LlamaIndexEmbeddingAdapter(Settings.embed_model),
            metadata={"hnsw:construction_ef": 100} )
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # DBにアクセス
        self.index = VectorStoreIndex.from_vector_store(vector_store=vector_store)

        TEXT_QA_SYSTEM_PROMPT = ChatMessage(
            content=(
                    "あなたは世界中で信頼されているQAシステムです。\n"
                    "事前知識ではなく、常に提供されたコンテキスト情報を使用してクエリに回答してください。\n"
                    "従うべきいくつかのルール:\n"
                    "1. 回答内で指定されたコンテキストを直接参照しないでください。\n"
                    "2. 「コンテキストに基づいて、...」や「コンテキスト情報は...」、またはそれに類するような記述は避けてください。"
                ),
            role=MessageRole.SYSTEM,
            )

        # QAプロンプトテンプレートメッセージ
        TEXT_QA_PROMPT_TMPL_MSGS = [
                                        TEXT_QA_SYSTEM_PROMPT,
                                        ChatMessage(
                                                        content=(
                                                                    "コンテキスト情報は以下のとおりです。\n"
                                                                    "---------------------\n"
                                                                    "{context_str}\n"
                                                                    "---------------------\n"
                                                                    "事前知識ではなくコンテキスト情報を考慮して、クエリに答えます。\n"
                                                                    "Query: {query_str}\n"
                                                                    "Answer: "
                                                                ),
                                                        role=MessageRole.USER,
                                                    ),
                                    ]

        # チャットQAプロンプト
        CHAT_TEXT_QA_PROMPT = ChatPromptTemplate(message_templates=TEXT_QA_PROMPT_TMPL_MSGS)
        # チャットRefineプロンプトテンプレートメッセージ
        CHAT_REFINE_PROMPT_TMPL_MSGS = [
                                            ChatMessage(
                                                            content=(
                                                                        "あなたは、既存の回答を改良する際に2つのモードで厳密に動作するQAシステムのエキスパートです。\n"
                                                                        "1. 新しいコンテキストを使用して元の回答を**書き直す**。\n"
                                                                        "2. 新しいコンテキストが役に立たない場合は、元の回答を**繰り返す**。\n"
                                                                        "回答内で元の回答やコンテキストを直接参照しないでください。\n"
                                                                        "疑問がある場合は、元の答えを繰り返してください。"
                                                                        "New Context: {context_msg}\n"
                                                                        "Query: {query_str}\n"
                                                                        "Original Answer: {existing_answer}\n"
                                                                        "New Answer: "
                                                                    ),
                                                            role=MessageRole.USER,
                                                        )
                                        ]

        # チャットRefineプロンプト
        CHAT_REFINE_PROMPT = ChatPromptTemplate(message_templates=CHAT_REFINE_PROMPT_TMPL_MSGS)



        print(messages)
        print(user_message)

        query_engine = self.index.as_query_engine(
                                                    streaming=True,
                                                    similarity_top_k=5,
                                                    text_qa_template=CHAT_TEXT_QA_PROMPT,
                                                    refine_template=CHAT_REFINE_PROMPT,
                                                    response_mode="refine"
                                                  )
        response = query_engine.query(user_message)

        return response.response_gen
