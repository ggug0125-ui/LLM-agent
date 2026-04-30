from pathlib import Path
import os

import pymupdf
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

pdf_path = Path(__file__).resolve().parent / "TOP5.pdf"

doc = pymupdf.open(pdf_path)

pdf_text = ""
for page in doc:
    pdf_text += page.get_text()

# PDF text can be too long for one request, so this test uses only the beginning.
pdf_text = pdf_text[:6000]

response = client.chat.completions.create(
    model="gpt-4o",
    temperature=0.2,
    messages=[
        {
            "role": "system",
            "content": "너는 PDF 문서를 읽고 핵심 내용을 쉽게 설명하는 도우미야.",
        },
        {
            "role": "user",
            "content": f"다음 PDF 내용을 한국어로 핵심만 5줄로 요약해줘.\n\n{pdf_text}",
        },
    ],
)

print(response.choices[0].message.content)
