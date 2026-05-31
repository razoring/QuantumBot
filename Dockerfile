FROM python:3.11-slim

WORKDIR /app

# Install dependencies required for psycopg2 and building Python packages
RUN apt-get update && apt-get install -y \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bot directory and README
COPY bot/ /app/bot/
COPY README.md /app/README.md

# Set Python path so imports work correctly
ENV PYTHONPATH="/app"

CMD ["python", "bot/instance.py"]
