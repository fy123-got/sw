FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    fonts-wqy-zenhei \
    libfontconfig1 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf2.0-0 \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p data charts logs

EXPOSE 12245

CMD ["python", "final_app.py", "--port", "12245"]
