FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create writable directories
RUN mkdir -p /tmp/.streamlit && chmod 777 /tmp/.streamlit
RUN mkdir -p /tmp/.cache && chmod 777 /tmp/.cache

# Set all Streamlit paths to /tmp (writable)
ENV HOME=/tmp
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV XDG_CONFIG_HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache

# Server config
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true

# Disable telemetry
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Expose port
EXPOSE 7860

# Run Streamlit
CMD streamlit run rag_ui.py --server.port=7860 --server.address=0.0.0.0 --server.headless=true
