FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories with proper permissions
RUN mkdir -p /.streamlit && chmod 777 /.streamlit
RUN mkdir -p /app/.streamlit && chmod 777 /app/.streamlit
RUN mkdir -p /.cache && chmod 777 /.cache

# Set environment variables
ENV STREAMLIT_HOME=/app/.streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 7860

# Run Streamlit
CMD streamlit run rag_ui.py --server.port=7860 --server.address=0.0.0.0
