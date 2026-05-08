FROM postgres:16

ARG PGVECTOR_VERSION=0.8.2

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
        wget \
        ca-certificates \
        build-essential \
        postgresql-server-dev-16 \
        locales && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && \
    locale-gen en_US.UTF-8 && \
    rm -rf /var/lib/apt/lists/*

ENV LANG=en_US.UTF-8
ENV LC_ALL=en_US.UTF-8

RUN wget -q https://github.com/pgvector/pgvector/archive/refs/tags/v${PGVECTOR_VERSION}.tar.gz && \
    tar -xzf v${PGVECTOR_VERSION}.tar.gz && \
    cd pgvector-${PGVECTOR_VERSION} && \
    make && \
    make install && \
    cd / && \
    rm -rf pgvector-${PGVECTOR_VERSION} v${PGVECTOR_VERSION}.tar.gz

COPY ./init.sql /docker-entrypoint-initdb.d/