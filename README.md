# RAG Notes Chatbot (FAISS)

Chat with your PDF notes using a fully local RAG pipeline. No OpenAI, no paid APIs, no cloud services.

## Description

This project lets you ask natural-language questions about a PDF of notes, such as "What is FAISS?" or "Why is chunking useful in RAG?", and get answers grounded only in the PDF's content. You can also ask follow-up questions like "Is it a vector store or a vector database?" and the chatbot understands what "it" refers to.

It is a Retrieval-Augmented Generation (RAG) pipeline. The PDF is split into chunks and converted to vector embeddings, which are stored in a FAISS index. A question is embedded the same way and compared with the stored vectors using cosine similarity. The 3 most relevant chunks are passed to a local Mistral model, which writes the answer using only that context. Everything runs on your own machine.

## Features

- Runs 100% locally: no API keys and no internet needed after setup
- Works with any text-based PDF
- Cosine-similarity semantic search with Top-K = 3
- FAISS index saved to disk and reloaded without re-embedding
- Retrieval verified two ways: manual NumPy cosine similarity and FAISS search return the same top chunks
- Answers are limited to the notes context, so the model says "I couldn't find that in the notes." for missing information
- Streaming answers in real time
- Shows source chunks and page numbers for every answer
- Conversational memory: follow-up questions are rewritten into standalone questions using the last 3 question-and-answer pairs
- Notebook-based, so each pipeline step can be run, inspected and fixed on its own
- Includes matplotlib visualisation of the query and the top 3 chunks (2D PCA view)

## How it works

```
INGESTION (ingest.ipynb, run once)
PDF -> PyPDFLoader -> Documents -> text cleaning -> RecursiveCharacterTextSplitter (500 / 80)
    -> Nomic Embed Text (via Ollama) -> FAISS (cosine, saved on disk)

QUERY + GENERATION (chat.ipynb, every question)
Question -> Nomic embedding -> cosine search in FAISS -> top 3 chunks
         -> prompt (context + question) -> Mistral (via Ollama) -> streamed answer + sources

CONVERSATIONAL RAG (chat_history.ipynb)
Follow-up + chat history -> Mistral rewrites it as a standalone question
         -> retrieve top 3 chunks -> prompt (context + history + question) -> Mistral -> answer
```

## Tech stack

| Component       | Tool                                                              |
| --------------- | ----------------------------------------------------------------- |
| Language        | Python 3.12                                                       |
| Framework       | LangChain                                                         |
| LLM             | Mistral (via Ollama)                                              |
| Embeddings      | Nomic Embed Text (via Ollama, 768 dimensions)                     |
| Vector store    | FAISS (cosine similarity via normalized inner product)            |
| PDF parsing     | PyPDF                                                             |
| Text splitting  | RecursiveCharacterTextSplitter (chunk_size=500, chunk_overlap=80) |
| Visualisation   | matplotlib, scikit-learn (PCA)                                    |

## Prerequisites

- Python 3.12
- [Ollama](https://ollama.com)
- About 8 GB of RAM (Mistral 7B) and about 5 GB of free disk space for the models
- VS Code with the Python and Jupyter extensions (or JupyterLab)

## Setup

### 1. Clone the repository

```
git clone https://github.com/<your-username>/RAG-Chatbot-2-FAISS.git
cd RAG-Chatbot-2-FAISS
```

### 2. Install the Ollama models

Make sure Ollama is running, then:

```
ollama pull mistral
ollama pull nomic-embed-text
ollama list
```

Both models should appear in the list.

### 3. Create the virtual environment and install packages

```
python -m venv .venv
```

Activate it:

```
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Install the packages:

```
pip install -r requirements.txt
```

### 4. Add your PDF

Copy your PDF into the project folder (next to the notebooks) and name it `RAG_Notes_Complete.pdf`. It must contain selectable text, not a scanned image.

### 5. Run the notebooks

Open the folder in VS Code (`code .`). In each notebook, click **Select Kernel**, then **Python Environments**, and choose the `.venv` of this project.

1. **`ingest.ipynb`** : run all cells from top to bottom. This loads the PDF, cleans it, splits it, embeds the chunks and stores them in `faiss_index/`. Run it once, and again whenever the PDF changes.
2. **`chat.ipynb`** : restart the kernel, then run all cells from top to bottom. Change the `question` variable and re-run to ask something else, or use `stream_answer()` and `stream_answer_with_sources()` in the last cells.
3. **`chat_history.ipynb`** : restart the kernel, then run all cells. Use `chat("your question")` to ask questions. Follow-ups work because the last 3 question-and-answer pairs are remembered.

## Using your own PDF

1. Put your PDF in the project folder.
2. In `ingest.ipynb`, Cell 2, set the file name:

   ```
   PDF_PATH = BASE_DIR / "your-file-name.pdf"
   ```

3. In `chat.ipynb` and `chat_history.ipynb`, edit the system prompt and replace "RAG (Retrieval-Augmented Generation)" with the topic of your document.
4. Re-run `ingest.ipynb` completely, then restart and re-run the chat notebooks.

## Configuration

| Setting               | Where        | Default            | Meaning                                              |
| --------------------- | ------------ | ------------------ | ---------------------------------------------------- |
| `PDF_PATH`            | ingest       | `RAG_Notes_Complete.pdf` | The document to index                          |
| `CHUNK_SIZE`          | ingest       | 500                | Maximum characters per chunk                         |
| `CHUNK_OVERLAP`       | ingest       | 80                 | Characters repeated between neighbouring chunks      |
| `FAISS_DIR`           | all          | `faiss_index`      | Folder where the index is saved and loaded           |
| `EMBED_MODEL`         | all          | `nomic-embed-text` | Embedding model (must match in all notebooks)        |
| `LLM_MODEL`           | chat         | `mistral`          | The chat model                                       |
| `TOP_K`               | chat         | 3                  | Number of chunks retrieved per question              |
| `MAX_HISTORY_MESSAGES`| chat_history | 6                  | Messages remembered (3 question-and-answer pairs)    |
| `temperature`         | chat         | 0                  | 0 gives factual, repeatable answers                  |

If you change the chunk settings, the PDF or the embedding model, re-run `ingest.ipynb`.

## Project structure

```
RAG-Chatbot-2-FAISS/
├── ingest.ipynb          # load -> clean -> split -> embed -> store in FAISS
├── chat.ipynb            # query -> embed -> retrieve -> prompt -> Mistral
├── chat_history.ipynb    # conversational RAG with query rewriting and memory
├── RAG_Notes_Complete.pdf
├── requirements.txt
├── assets/               # demo GIF and screenshots
└── faiss_index/          # generated by ingest.ipynb (not in Git)
```

## Troubleshooting

| Problem                            | Fix                                                                                  |
| ---------------------------------- | ------------------------------------------------------------------------------------ |
| `Connection refused` on port 11434 | Ollama isn't running. Start the Ollama app or run `ollama serve`                     |
| `model "mistral" not found`        | Run `ollama pull mistral`                                                            |
| `PDF exists: False`                | Check the file name in `PDF_PATH` and that the notebook runs from the project folder |
| No text extracted from the PDF     | The PDF is a scanned image. Use a text-based PDF                                     |
| `Index folder exists: False`       | Run `ingest.ipynb` first                                                             |
| `ModuleNotFoundError`              | The notebook isn't using the project's `.venv`. Re-select the kernel                 |
| `Normalizing L2 is not applicable` warning | Harmless. The vectors are already normalized when the index is built         |
| Very slow answers                  | Normal on CPU, especially the first question while Mistral loads                     |

## Known limitations

- Very short chunks (such as a page header only) can appear in the top results and add noise
- A small local model can sometimes add general knowledge beyond the retrieved context
- The index is small (20 chunks), so similarity scores are modest

## Security note

The FAISS index is loaded with `allow_dangerous_deserialization=True`. Only load index files that you created yourself.

## Possible improvements

- Remove tiny chunks and repeated page headers before indexing
- Add a score threshold, MMR retrieval or hybrid search (BM25 + dense)
- Add multi-query retrieval or reranking
- Wrap the chat in a Streamlit, Gradio or FastAPI app

## License

MIT
