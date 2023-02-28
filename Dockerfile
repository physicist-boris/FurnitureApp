FROM python:3.7-stretch

WORKDIR /usr/src/app
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN pip install --no-cache-dir --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt --no-cache-dir
COPY . .
EXPOSE 80
EXPOSE 443
CMD uvicorn main:app --host 0.0.0.0 --port 80