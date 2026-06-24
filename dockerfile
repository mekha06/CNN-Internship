FROM python:3.11

WORKDIR /app

COPY requirements-deploy.txt .

RUN pip install --upgrade pip

RUN pip install \
    --index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1 torchvision==0.20.1

RUN pip install --no-cache-dir -r requirements-deploy.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]