# ベースイメージ(CUDA)の指定
FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# 必要なパッケージをインストール
RUN apt-get update && apt-get install -y python3-pip python3-venv git nano curl

# RUN curl -fsSL https://ollama.com/install.sh | sh

# 作業ディレクトリを設定
WORKDIR /app

COPY ./llamaindexer/code /app

# Python仮想環境の作成
# RUN python3 -m venv /app/.venv

# 仮想環境をアクティベートするコマンドを.bashrcに追加
# RUN echo "source /app/.venv/bin/activate" >> /root/.bashrc

# JupyterLabのインストール
# RUN /app/.venv/bin/pip install Jupyter jupyterlab

# LlamaIndex関連のインストール
RUN pip3 install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
RUN pip3 install -r requirement.txt


# コンテナの起動時にbashを実行
CMD ["/bin/bash"]

# CMD ["python3", "get_index_for_RAG.py"]