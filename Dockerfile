# For platforms that deploy from a container (Fly.io, Hugging Face Spaces,
# Google Cloud Run). Render does not need this -- it builds from
# requirements.txt directly.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py datasource.py indicators.py crosssection.py ./

ENV MCP_TRANSPORT=http
ENV PORT=8000
EXPOSE 8000

# Most platforms inject PORT; server.py reads it.
CMD ["python", "server.py"]
