import os
from pathlib import Path
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from crewai import LLM
from crewai.tools import BaseTool

load_dotenv()

llm = LLM(
    model=os.getenv("OPENROUTER_MODEL"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url=os.getenv("OPENROUTER_BASE_URL")
)

# llm = LLM(
#     model="ollama/tinydolphin:latest",
#     base_url="http://localhost:11434"
# )


class DocumentKnowledgeTool(BaseTool):
    name: str = "document_knowledge_tool"
    description: str = "Extract and summarize files from configured directory (PDF or TXT)."
    directory: str = ""

    def __init__(self, directory=None, **kwargs):
        super().__init__(**kwargs)
        self.directory = Path(directory or Path(__file__).resolve().parents[3] / "uploads")

    def _run(self, **kwargs) -> str:
        """Extract text and summarize key points using the given LLM."""

        valid_files = [f for f in os.listdir(self.directory) if f.lower().endswith((".pdf", ".txt"))]
        if not valid_files:
            return "No valid uploads (PDF/TXT) found."

        file_path = self.directory / valid_files[0]
        text = ""

        try:
            if file_path.suffix.lower() == ".pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    text += page.extract_text() or ""
            elif file_path.suffix.lower() == ".txt":
                text = file_path.read_text(encoding="utf-8")
            else:
                return "Unsupported file type."
        except Exception as e:
            return f"Error reading file: {str(e)}"

        if not text.strip():
            return "No readable text found in the file."

        if llm:
            prompt = (
                "You are an expert content summarizer. "
                "Extract key facts, entities, and important ideas from the following text:\n\n"
                f"{text[:6000]}"
                "\n\nProvide a concise factual summary."
            )
            return llm.call(prompt)

        return text[:2000] + "\n\n(Text truncated — no summarization performed.)"