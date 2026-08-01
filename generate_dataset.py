"""
generate_dataset.py

Calls a teacher LLM (any OpenAI-compatible endpoint: vLLM, Together, Groq,
Fireworks, etc.) once per chunk and asks it to produce ALL FIVE training-pair
variations in a single structured JSON response. This avoids 5x the API calls
and resolves the "one chunk at a time but needs noise from elsewhere"
contradiction by explicitly passing a couple of decoy chunks alongside the
target chunk.

Usage:
    pip install openai --break-system-packages
    export TEACHER_API_BASE="http://localhost:8000/v1"     # vLLM OpenAI server, or Together/Groq base url
    export TEACHER_API_KEY="sk-..."                        # "EMPTY" is fine for local vLLM
    export TEACHER_MODEL="meta-llama/Llama-3.3-70B-Instruct"

    python generate_dataset.py \
        --chunks chunks.jsonl \
        --event_metadata event_metadata.txt \
        --college_name "Sahyadri College of Engineering and Management" \
        --output train.jsonl
"""

import argparse
import json
import os
import random
import re
import time

from openai import OpenAI

SYSTEM_PROMPT_TEMPLATE = """You are an advanced AI data synthesis engine. You generate fine-tuning \
training data to teach a SMALL model to read noisy RAG prompts and to refuse \
gracefully instead of hallucinating.

You will be given:
- [TARGET CHUNK]: the real text the question/answer must be grounded in.
- [DECOY CHUNKS]: unrelated text from other parts of the paper collection, \
used only to add noise around the target chunk for Variation 1.
- [EVENT METADATA]: facts about the {college_name} conference, used only for Variation 5.

Generate exactly ONE JSON object (not JSON Lines) with a top-level key \
"pairs" containing a list of 5 objects, one per variation below, each with \
keys "variation", "context", "question", "answer".

Variation 1 - High Noise / Attention Lock:
  Take one crucial engineering metric/result from [TARGET CHUNK]. Build a \
  "context" field by interleaving 1-2 sentences from [DECOY CHUNKS] before \
  and after the sentence containing that metric, so the metric is buried in \
  irrelevant surrounding text. The question must target that exact metric. \
  The answer must isolate and state ONLY that metric, factually, ignoring the noise.

Variation 2 - Direct Q&A / Plain English:
  Use [TARGET CHUNK] as "context" verbatim (or a trimmed version of it). Ask \
  a standard technical question about the methodology or results, and give a \
  clear, jargon-free, comprehensive answer.

Variation 3 - Truthfulness / Refusal:
  Use [TARGET CHUNK] as "context". Ask a question that SOUNDS like it belongs \
  to this paper's topic but whose answer is NOT present in [TARGET CHUNK]. \
  The answer must be EXACTLY: "I am sorry, but the provided context does not \
  contain information to answer this question."

Variation 4 - Layman ELI5:
  Use [TARGET CHUNK] as "context". Ask "Explain what this paper built, in 2 \
  simple sentences, for a non-technical audience." Answer in max 2 short \
  sentences, no jargon.

Variation 5 - Event Memorization:
  Use [EVENT METADATA] as "context" (paraphrase it slightly so wording isn't \
  identical every time). Ask a natural question about the conference (dates, \
  location, chairs, topics, etc.) and answer it directly from the metadata.

Output ONLY the raw JSON object. No markdown fences, no commentary, no \
leading/trailing text.
"""

USER_PROMPT_TEMPLATE = """[TARGET CHUNK]
{target_chunk}

[DECOY CHUNKS]
{decoy_chunks}

[EVENT METADATA]
{event_metadata}
"""


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def build_chatml_record(college_name: str, context: str, question: str, answer: str):
    return {
        "messages": [
            {
                "role": "system",
                "content": f"You are an AI assistant specialized in the IEEE Conference papers hosted at {college_name}.",
            },
            {
                "role": "user",
                "content": f"Context: {context}\n\nQuestion: {question}",
            },
            {"role": "assistant", "content": answer},
        ]
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True)
    ap.add_argument("--event_metadata", required=True, help="Plain text file with conference facts")
    ap.add_argument("--college_name", required=True)
    ap.add_argument("--output", default="train.jsonl")
    ap.add_argument("--max_retries", type=int, default=3)
    args = ap.parse_args()

    client = OpenAI(
        base_url=os.environ["TEACHER_API_BASE"],
        api_key=os.environ.get("TEACHER_API_KEY", "EMPTY"),
    )
    model_name = os.environ["TEACHER_MODEL"]

    chunks = load_jsonl(args.chunks)
    event_metadata = open(args.event_metadata, encoding="utf-8").read()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(college_name=args.college_name)

    out_f = open(args.output, "w", encoding="utf-8")
    total_written = 0

    for idx, rec in enumerate(chunks):
        target = rec["text"]

        # sample 2 decoy chunks from elsewhere in the corpus (different chunk, any paper)
        pool = [c for c in chunks if c["chunk_id"] != rec["chunk_id"]]
        decoys = random.sample(pool, k=min(2, len(pool)))
        decoy_text = "\n---\n".join(d["text"][:400] for d in decoys)

        user_prompt = USER_PROMPT_TEMPLATE.format(
            target_chunk=target,
            decoy_chunks=decoy_text,
            event_metadata=event_metadata,
        )

        parsed = None
        for attempt in range(args.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.7,
                )
                raw = resp.choices[0].message.content
                parsed = json.loads(strip_json_fences(raw))
                break
            except Exception as e:
                print(f"[chunk {idx}] attempt {attempt+1} failed: {e}")
                time.sleep(2)

        if parsed is None:
            print(f"[chunk {idx}] giving up after {args.max_retries} attempts")
            continue

        for pair in parsed.get("pairs", []):
            record = build_chatml_record(
                args.college_name,
                pair.get("context", ""),
                pair.get("question", ""),
                pair.get("answer", ""),
            )
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            total_written += 1

        if idx % 10 == 0:
            print(f"Processed {idx}/{len(chunks)} chunks, {total_written} pairs so far")

    out_f.close()
    print(f"Done. Wrote {total_written} training pairs -> {args.output}")


if __name__ == "__main__":
    main()
