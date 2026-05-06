import os
from pathlib import Path

import pymupdf


BASE_DIR = Path(__file__).resolve().parents[1]
pdf_file_path = BASE_DIR / "data" / "농업 현장데이터의 디지털 전환을 위한 영농일지 표준화와 자동화 전략.pdf"

doc = pymupdf.open(pdf_file_path)

header_height = 80
footer_height = 80

full_text = ""

for page in doc:
    rect = page.rect

    text = page.get_text(
        clip=(0, header_height, rect.width, rect.height - footer_height)
    )

    full_text += text + "\n------------------------------------\n"

pdf_file_name = os.path.basename(pdf_file_path)
pdf_file_name = os.path.splitext(pdf_file_name)[0]

txt_file_path = BASE_DIR / "output" / f"{pdf_file_name}_with_preprocessing.txt"

with open(txt_file_path, "w", encoding="utf-8") as f:
    f.write(full_text)
