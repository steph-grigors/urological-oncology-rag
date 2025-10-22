FROM python:3.11-slim

# Run as root initially
USER root

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all app files
COPY . .

# Create writable directories in /tmp
RUN mkdir -p /tmp/.streamlit && \
    mkdir -p /tmp/.cache && \
    chmod -R 777 /tmp

# Set all environment variables to use /tmp
ENV HOME=/tmp
ENV STREAMLIT_HOME=/tmp/.streamlit
ENV STREAMLIT_CONFIG_DIR=/tmp/.streamlit
ENV XDG_CONFIG_HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache
ENV MPLCONFIGDIR=/tmp/matplotlib
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose port
EXPOSE 7860

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health || exit 1

# Run app (staying as root to avoid permission issues)
CMD ["streamlit", "run", "rag_ui.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.headless=true"]
