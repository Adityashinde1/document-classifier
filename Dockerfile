FROM ubuntu:22.04

WORKDIR /app

COPY . /app

ENV TZ=Europe/Paris

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install git curl unzip python3 python3-pip -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && unzip awscliv2.zip && ./aws/install
 
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt && pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu && python3 -m pip install 'git+https://github.com/facebookresearch/detectron2.git' && apt install tesseract-ocr -y

CMD ["python3", "app.py"]