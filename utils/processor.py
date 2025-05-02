from openai import OpenAI
import fitz
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from openai import RateLimitError, APIError

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@retry(
    wait=wait_random_exponential(min=1, max=10),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((RateLimitError, APIError))
)
def ask_openai(prompt):
    """Send prompt to OpenAI with retry logic."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3
    )
    return response.choices[0].message.content

def extract_text(pdf_path):
    """Extract raw text from all pages of a PDF."""
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

def chunk_text(text, max_tokens=2500):
    """Split long text into manageable chunks."""
    paragraphs = text.split("\n\n")
    chunks, current_chunk = [], []
    current_len = 0
    for para in paragraphs:
        current_len += len(para)
        current_chunk.append(para)
        if current_len >= max_tokens:
            chunks.append("\n\n".join(current_chunk))
            current_chunk, current_len = [], 0
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks

def summarize_audit(text):
    """Generate a compact bullet-point summary of an audit."""
    chunks = chunk_text(text)
    partial_summaries = []
    for chunk in chunks:
        prompt = (
            "You are an expert auditor. Provide a compact summary (max 6 bullet points) "
            "for the following audit section including only the most important:\n"
            "- Key Findings\n- Measures Taken\n- Issues or Outcomes\n\n"
            f"AUDIT SECTION:\n{chunk}"
        )
        partial_summaries.append(ask_openai(prompt))
    return "\n".join(partial_summaries)

def compare_audits(summaries):
    """Compare multiple audits to find similarities and differences."""
    joined = "\n\n".join(f"Audit {i+1}:\n{summary}" for i, summary in enumerate(summaries))
    prompt = (
        "Compare the following audit summaries and provide a structured comparison with:\n"
        "- Common findings\n- Differences in measures or risks\n- Shared problems\n\n"
        f"{joined}"
    )
    return ask_openai(prompt)

def extract_learnings(summaries):
    """Extract 3–5 general learnings or best practices from summaries."""
    joined = "\n\n".join(summaries)
    prompt = (
        "Based on the following audit summaries, list 3–5 key lessons or audit best practices "
        "for future use:\n\n"
        f"{joined}"
    )
    return ask_openai(prompt)

def analyze_audits(paths, filenames):
    """Process multiple audits and return all results and exportable text."""
    summaries = []
    for path, name in zip(paths, filenames):
        text = extract_text(path)
        summary = summarize_audit(text)
        summaries.append(f"### Audit of {name}\n\n{summary}")

    comparison = compare_audits(summaries)
    learnings = extract_learnings(summaries)

    full_text = "=== AUDIT SUMMARIES ===\n\n" + "\n\n".join(summaries)
    full_text += "\n\n=== COMPARISON ===\n\n" + comparison
    full_text += "\n\n=== LEARNINGS ===\n\n" + learnings

    return summaries, comparison, learnings, full_text
