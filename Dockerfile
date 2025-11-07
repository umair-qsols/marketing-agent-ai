# ---- Base image -------------------------------------------------
FROM python:3.11-slim

# ---- Environment ------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=off \
    PIP_DISABLE_PIP_VERSION_CHECK=on \
    PORT=8501

# ---- System packages --------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential gcc libssl-dev libffi-dev \
        libxml2-dev libxslt1-dev zlib1g-dev curl git && \
    rm -rf /var/lib/apt/lists/*

# ---- Non-root user ---------------------------------------------
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME

# ---- Workdir ----------------------------------------------------
WORKDIR /app

# ---- Install Python deps ----------------------------------------
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---- Copy source (with correct ownership) -----------------------
COPY --chown=$USERNAME:$USERNAME . .

# ---- Ensure required folders exist -------------------------------
RUN mkdir -p ./templates ./chroma_db .streamlit

# ---- Use non-root user ------------------------------------------
USER $USERNAME

# ---- Expose & healthcheck ---------------------------------------
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# ---- Run ---------------------------------------------------------
CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0"]