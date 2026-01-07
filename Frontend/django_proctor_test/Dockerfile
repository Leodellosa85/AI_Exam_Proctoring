# =========================
# Stage 1: Builder
# =========================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Build dependencies ONLY here
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        --prefix=/install \
        -r requirements.txt

# COPY requirements-prod.txt .

# RUN pip install --upgrade pip && \
#     pip install --no-cache-dir \
#         --target=/install \
#         -r requirements-prod.txt \
#         -f https://download.pytorch.org/whl/torch_stable.html


# =========================
# Stage 2: Runtime
# =========================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# ENV PYTHONPATH=/usr/local/lib/python3.11/site-packages


WORKDIR /app

# Runtime-only deps (NO gcc)
RUN apt-get update && apt-get install -y \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# Copy python deps from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY . .

# Collect static files
RUN python manage.py collectstatic --noinput

EXPOSE 8002

CMD ["daphne", "-b", "0.0.0.0", "-p", "8002", "django_proctor_test.asgi:application"]
