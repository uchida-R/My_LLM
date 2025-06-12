FROM ollama/ollama:latest


COPY ./Ollama/entrypoint.sh /root/entrypoint.sh
RUN chmod +x /root/entrypoint.sh

ENV LLM_MODEL="gemma3:12b"

ENTRYPOINT ["/root/entrypoint.sh"]


