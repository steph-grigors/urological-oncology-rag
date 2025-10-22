FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create writable directories
RUN mkdir -p /tmp/.streamlit && chmod 777 /tmp/.streamlit

# Set environment variables
ENV HOME=/tmp
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# CRITICAL: Expose port 7860
EXPOSE 7860

# Start Streamlit
CMD ["streamlit", "run", "rag_ui.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
