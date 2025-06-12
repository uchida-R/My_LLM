from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import StorageContext
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.schema import TextNode, TransformComponent

from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import SemanticSplitterNodeParser
from llama_index.core.utils import get_cache_dir
from llama_index.core.extractors import TitleExtractor,KeywordExtractor,QuestionsAnsweredExtractor
import chromadb
from chromadb import EmbeddingFunction, Documents, Embeddings

import os
from dotenv import load_dotenv
import unicodedata
import re
import nltk
import os
    

load_dotenv()
LLM_MODEL = os.getenv("LLM_MODEL")
EMBED_MODEL = os.getenv("EMBED_MODEL")

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")

chroma_db_hostname = os.getenv('VECDB_HOSTNAME', 'localhost')
chroma_db_port = int(os.getenv('VECDB_PORT', '8000'))
data_folder = os.getenv('IMP_DATA_FOLDER')

def split_by_sentence_tokenizer():
    cache_dir = get_cache_dir()
    nltk_data_dir = os.environ.get("NLTK_DATA", cache_dir)

    # update nltk path for nltk so that it finds the data
    if nltk_data_dir not in nltk.data.path:
        nltk.data.path.append(nltk_data_dir)

    sent_detector = nltk.RegexpTokenizer(u'[^　！？。]*[！？。.\n]')
    return sent_detector.tokenize       


# Ollamaサーバのllmを使い、ollamaサーバーで処理してもらう
Settings.llm = Ollama(model = LLM_MODEL,request_timeout=1000.0, context_window=8192,base_url=OLLAMA_HOST)
# HuggingFaceの埋め込みモデルを使う 埋め込みモデルはchromadbとllamaindexで統一すべき
Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL,embed_batch_size=8,max_length=512)
# クソ長いテキストを意味単位でnodeに分割する 引数はよくわからん
Settings.node_parser = SemanticSplitterNodeParser.from_defaults(
    buffer_size=1, breakpoint_percentile_threshold=95, embed_model=Settings.embed_model
)



# chromadbサーバーと接続
chroma_client = chromadb.HttpClient(host=chroma_db_hostname, port=chroma_db_port)

# llamaindexの埋め込みモデルをchromadbのEmbeddingFunctionに変換する
class ChromaDB_LlamaIndexEmbeddingAdapter(EmbeddingFunction):
    def __init__(self, ef: BaseEmbedding):
        self.ef = ef

    def __call__(self, input: Documents) -> Embeddings:
        return [node.embedding for node in
                self.ef([TextNode(text=doc) for doc in input])]

# chroma_client.delete_collection(name="labs_rag") # deleate collection if exists デバッグ用
collection = chroma_client.get_or_create_collection(
    name="labs_rag", # コレクション名 コレクション名ごとにデータを分けることができる
    embedding_function=ChromaDB_LlamaIndexEmbeddingAdapter(Settings.embed_model),
    metadata={"hnsw:construction_ef": 100} )
    #metadata = {"hnsw:space": "cosine"}) # デフォルトはcosine距離でベクトルを比較する これだとデータの追加ができないらしい 他にもいっぱいある

#pdfとパワポのファイルを処理する
required_exts = [".pdf", ".pptx"]

# パワポを処理するときにGPT-2とViTを使っている -> できるだけ統一したい
reader = SimpleDirectoryReader(
    input_dir=data_folder,
    required_exts=required_exts,
    recursive=False, # Trueにするとサブディレクトリも再帰的に読み込む
    filename_as_id=True # ドキュメント固有のIDをファイル名にする NASには同じ名前のファイルはないからこれで問題ないはず
)

# よくわからん おまじない
vector_store = ChromaVectorStore(chroma_collection=collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)


# テキストの含まれる半角カナを全角カナ、全角数字を半角数字などに変換。
class TextNormalizer(TransformComponent):
    def __call__(self, nodes, **kwargs):
        for node in nodes:
            node.set_content(unicodedata.normalize('NFKC', node.get_content()))
            # node.set_content(unicodedata.normalize('NFKC', node.get_content().replace("\n", ""))) #改行を削除したい場合
        return nodes


# テキストのタイトルやキーワードを抽出する
pipeline = IngestionPipeline(
    transformations=[
        Settings.node_parser,
        TextNormalizer(),
        TitleExtractor(),
        KeywordExtractor(keywords=10),

        # QuestionsAnsweredExtractor(questions=3) # 質問応答の抽出はあり得ないほど時間がかかるのでコメントアウト
    ]
)

# dbから消すべきデータと追加すべきデータのファイル名を取得する関数
# docsはSimpleDirectoryReaderで読み込んだドキュメントのリスト(をpipelineで処理したもの)
# collectionはChromaVectorStoreのコレクション
# 消すべきものの条件: dbには存在するがdocsに存在しない、または更新日が異なる(古いと思われる)データ
# 追加すべきものの条件: docsには存在するがdbに存在しない、または更新日が異なる(新しいと思われる)データ
# 全部読み出しているのでアホ時間かかるうえにメモリに限界がくる　後回し
def get_sync_file_names_and_dates(docs, collection):
    # docsからファイル名→更新日辞書を作成
    file_dict = {}
    for doc in docs:
        dict_meta = dict(s.split(": ", 1) for s in doc.get_metadata_str().split("\n"))
        file_name = dict_meta["file_name"]
        last_modified_date = dict_meta["last_modified_date"]
        # 先に出てきたものを優先（重複排除）
        if file_name not in file_dict:
            file_dict[file_name] = last_modified_date

    # DBからファイル名→更新日辞書を作成
    db_dict = {}
    for meta in collection.get()["metadatas"]:
        if meta is not None:
            db_file_name = meta.get('file_name')
            db_last_modified_date = meta.get('last_modified_date')
            if db_file_name and (db_file_name not in db_dict):
                db_dict[db_file_name] = db_last_modified_date

    # set化して高速比較
    file_names_set = set(file_dict)
    db_file_names_set = set(db_dict)

    # 差分計算（O(N)）
    delete_file_names = [db_file for db_file in db_file_names_set
                         if db_file not in file_names_set or file_dict.get(db_file) != db_dict[db_file]]
    add_file_names = [file for file in file_names_set
                      if file not in db_file_names_set or db_dict.get(file) != file_dict[file]]

    return delete_file_names, add_file_names

# docsからfile_nameを取得し、追加すべきものだけを抽出する
def filter_func(docs, add_file_names):
    # add_file_namesをsetにして高速化
    add_file_names_set = set(add_file_names)
    new_docs = []
    for doc in docs:
        dict_meta = dict(s.split(": ", 1) for s in doc.get_metadata_str().split("\n"))
        
        file_name = dict_meta["file_name"]
        if file_name in add_file_names_set:
            if "page_label" not in dict_meta:
                doc.doc_id = f"{file_name}"
            else:
                page_label = dict_meta["page_label"]
                doc.doc_id = f"{file_name}_part{page_label}"
            new_docs.append(doc)
    return new_docs

# 消すべきデータをコレクションから削除する関数
def delete_files_from_collection(collection, delete_file_names):
    """
    delete_file_namesに含まれるfile_nameを持つ全データをcollectionから削除する
    """
    for file_name in delete_file_names:
        # file_nameが一致する全データを削除
        collection.delete(where={"file_name": file_name})
# print(collection.get()["ids"])
# print([meta.get('file_name') for meta in collection.get()["metadatas"] if meta is not None])
#print(len(collection.get(where={"$and": [{"document_id" : "2212.09748v2.pdf_part0"}, {"page_label": "1"},{'last_modified_date': '2025-06-04'}]})))
# print(collection.peek(1))


print("collection.count() = ", collection.count())
#ファイルの読み出し, ページへの分割(pptxはやってくれない)
docs = reader.load_data()
print(f"Loaded {len(docs)} docs")

delete_file_names, add_file_names = get_sync_file_names_and_dates(docs, collection)
print(f"delete_file_names: {delete_file_names}")
print(f"add_file_names: {add_file_names}")
if delete_file_names:
    delete_files_from_collection(collection, delete_file_names)
    print(f"Deleted {len(delete_file_names)} files from collection")
docs = filter_func(docs, add_file_names)
print(f"Filtered {len(docs)} docs")

nodes = pipeline.run(documents=docs) # 読み出したドキュメントの処理
index = VectorStoreIndex(
    nodes,
    storage_context=storage_context,
    embed_model=Settings.embed_model,
    show_progress=True,
) # indexの生成 chromadbへの保存もこのタイミングで行われる

print("collection.count() = ", collection.count())


# index = VectorStoreIndex.from_vector_store(vector_store=vector_store) # dbから読み出し

# query_engine = index.as_query_engine() # よくわかんにゃい
# response = query_engine.query("ディープフェイクテキストとは何ですか?")
# print("Response: ", response)




