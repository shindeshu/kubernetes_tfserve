FROM python:3.8.12-slim

WORKDIR /app

COPY gateway_requirements.txt  ./

RUN pip install fastapi==0.104.1 uvicorn==0.24.0 grpcio==1.42.0 numpy==1.24.4 tensorflow-protobuf==2.7.0 requests==2.31.0 Pillow==10.1.0

COPY gateway.py protobuf_utils.py ./

EXPOSE 9696

ENTRYPOINT ["python", "gateway.py"]