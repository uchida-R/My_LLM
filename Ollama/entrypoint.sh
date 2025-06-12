#!/bin/sh
set -e

# もし環境変数 LLM_MODEL が未設定ならエラーメッセージを出して終了
if [ -z "$LLM_MODEL" ]; then
  echo "Error: 環境変数 LLM_MODEL が設定されていません。"
  exit 1
fi

echo ">>> Starting ollama server in background..."
ollama serve &
SERVER_PID=$!

echo ">>> Waiting for ollama server to become ready..."
sleep 5

# モデルを pull。すでに取得済みならスキップされます。
echo ">>> Pulling model: $LLM_MODEL"
ollama pull "$LLM_MODEL"

# モデルが正常に取得できたか確認
if [ $? -ne 0 ]; then
  echo "Error: モデルのダウンロードに失敗しました: $LLM_MODEL"
  exit 1
fi

echo ">>> ollama server is up. Waiting in foreground (PID: $SERVER_PID)..."
wait "$SERVER_PID"
