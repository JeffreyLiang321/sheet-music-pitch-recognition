# Build the frontend, then a slim Python image that serves it with the ONNX models.
FROM node:20-alpine AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY omr ./omr
COPY app ./app
COPY models/*.onnx ./models/
COPY --from=web /web/dist ./web/dist

# Hugging Face Spaces expects the app on 7860
ENV PORT=7860 MAX_PAGES=6
EXPOSE 7860
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
