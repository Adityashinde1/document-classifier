FROM python:3.8-slim-buster

WORKDIR /app

COPY . /app

RUN pip install -r requirements.txt && pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu \
    python -m pip install 'git+https://github.com/facebookresearch/detectron2.git' && sudo apt install tesseract-ocr 

CMD ["python3", "app.py"]