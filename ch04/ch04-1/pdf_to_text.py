import pymupdf # pip install PyMuPDF
import os

#① pdf 파일을 페이지별로 내용을 읽어 온다.
pdf_file_path = "../../ch04/data/과정기반 작물모형을 이용한 웹 기반 밀 재배관리 의사결정 지원시스템 설계 및 구축.pdf"
doc = pymupdf.open(pdf_file_path)

full_text = ''

#② doc 객체를 반복하여 page변수에 넣고 full_text 문자열에 추가 기록한다.
for page in doc: # 문서 페이지 반복
    text = page.get_text() # 페이지 텍스트 추출
    full_text += text

#③ pdf 파일명을 추출한다.
pdf_file_name = os.path.basename(pdf_file_path)
pdf_file_name = os.path.splitext(pdf_file_name)[0] # 확장자 제거

#④ pdf 파일명에 txt 확장자를 붙이고 full_txt 변수 내용을 기록한다.
# cho4/output 폴더 추가 필수
txt_file_path = f"../../ch04/output/{pdf_file_name}.txt"
with open(txt_file_path, 'w', encoding='utf-8') as f:
    f.write(full_text)

# 결과 보기
# (venv) PS C:\Aiprojects> cd .\ch04\
# (venv) PS C:\Aiprojects\ch04> cd .\ch04-1\
# (venv) PS C:\Aiprojects\ch04\ch04-1> py .\pdf_to_text.py
# (venv) PS C:\Aiprojects\ch04\ch04-1> 
# output/과정~. txt를 확인 해보면 머리말과 꼬리말등이 포함되어 보인다.