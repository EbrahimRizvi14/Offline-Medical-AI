# Medical AI RAG Pipeline

A Python project for extracting structured content from medical PDFs and using it in a retrieval-augmented generation (RAG) workflow. The pipeline is designed to parse reports, preserve document structure such as headings, tables, and figures, build semantic chunks, retrieve relevant context with FAISS, and answer questions using a local LLM.

## Features

- Extracts native text from PDFs with PyMuPDF.
- Detects headings using font-size heuristics.
- Extracts tables and serializes them into Markdown-like text.
- Supports OCR for image-based content through Tesseract.
- Builds semantic chunks with page, section, token, and position metadata.
- Creates vector embeddings using `sentence-transformers`.
- Stores embeddings in an in-memory FAISS index.
- Generates answers and summaries through an Ollama model endpoint.

## Project Structure

```text
.
|-- data/                    # Local PDF/report files
|-- pdf_parser/
|   |-- chunker.py           # Semantic chunk creation and chunk merging logic
|   |-- formatter.py         # Normalizes extracted text, tables, and image OCR
|   |-- image_extractor.py   # Image OCR extraction with pytesseract
|   |-- models.py            # Dataclasses for parsed PDF elements
|   |-- table_extractor.py   # Table detection and Markdown serialization
|   |-- text_extractor.py    # Text extraction and heading detection
|   `-- utils.py             # Token counting, cleanup, IDs, bounding boxes
|-- rag/
|   `-- pipeline.py          # Embedding, FAISS retrieval, prompt, LLM calls
|-- engineering_flow.md      # Design notes for the parser and RAG flow
|-- main.py                  # Project entry point placeholder
|-- requirements.txt         # Python dependencies
`-- README.md
```

## Requirements

- Python 3.10 or newer recommended.
- Tesseract OCR installed locally if you want OCR support.
- Ollama running locally for answer generation.
- A downloaded Ollama model matching the code configuration, currently `gemma3:4b`.

The image extractor currently points to this Windows Tesseract path:

```python
C:\Users\Ebrahim\AppData\Local\Programs\Tesseract-OCR\tesseract.exe
```

Update `pdf_parser/image_extractor.py` if Tesseract is installed somewhere else.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install and start Ollama, then pull the configured model:

```bash
ollama pull gemma3:4b
ollama serve
```

## Usage

Place a PDF report in the `data/` folder, for example:

```text
data/report.pdf
```

The RAG pipeline entry point is in `rag/pipeline.py`:

```bash
python rag/pipeline.py
```

The sample script does two things:

- Runs question-answering mode for a hardcoded medical query.
- Runs summary mode over the first chunks of the document.

Example usage from Python:

```python
from rag.pipeline import RAGPipeline

rag = RAGPipeline("data/report.pdf")

answer = rag.run_pipeline("What diagnosis is mentioned in the assessment?")
summary = rag.summarize_document()

print(answer)
print(summary)
```

## RAG Flow

1. Extract PDF content with text, table, and image/OCR parsers.
2. Convert extracted content into structured content elements.
3. Merge related text into semantic chunks.
4. Embed chunks with `BAAI/bge-small-en-v1.5`.
5. Normalize vectors and index them with FAISS.
6. Retrieve relevant chunks for a user question.
7. Build a context-only prompt.
8. Send the prompt to the local Ollama generation endpoint.

## Notes

- This project is in active development.
- `rag/pipeline.py` currently assumes the chunking step returns plain text chunks suitable for embedding.
- The chunker in `pdf_parser/chunker.py` is built around `PageLayout` and `ContentElement` dataclasses, so parser-to-chunker integration may need alignment before the full end-to-end pipeline is production-ready.
- Generated medical answers should be treated as informational only and reviewed by a qualified clinician.

## Development Roadmap

- Connect the formatter output cleanly to the semantic chunker.
- Persist chunk metadata and FAISS indexes to disk.
- Add citations with page numbers and section headings.
- Add reranking before final context selection.
- Add tests for PDF extraction, chunking, retrieval, and prompt generation.
