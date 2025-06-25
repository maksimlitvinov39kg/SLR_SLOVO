FROM python:3.11

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements_d.txt ./

RUN pip install --upgrade pip

RUN pip install torch==2.0.0+cpu torchvision==0.15.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

RUN pip install --no-cache-dir -r requirements_d.txt

COPY service/ ./

RUN mkdir -p /tmp/streamlit && \
    mkdir -p /app/models

ENV PYTHONPATH=/app
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

RUN python -c "import os; required_files=['model.py', 'preprocessor.py', 'statistics.json']; missing=[f for f in required_files if not os.path.exists(f)]; print(f'Missing files: {missing}' if missing else 'All required files present')"

CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501", "--server.headless", "true"]