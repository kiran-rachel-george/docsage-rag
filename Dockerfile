FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
COPY src ./src
RUN pip install --no-cache-dir -e .
COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "docsage.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
