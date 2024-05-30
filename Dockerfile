# Use an NVIDIA CUDA base image
FROM nvidia/cuda:12.3.1-runtime-ubuntu20.04 as builder

# Avoid prompts from apt and combine update/install/cleanup steps
ENV DEBIAN_FRONTEND=noninteractive

# Install OS dependencies, including Git, Python, and cleanup in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ca-certificates \
    dos2unix \
    git \
    software-properties-common \
    python3.9 \
    python3-pip \
    && ln -sf /usr/bin/python3.9 /usr/bin/python \
    && ln -sf /usr/bin/python3.9 /usr/bin/python3 \
    && python3.9 -m pip install --upgrade pip \
    && rm -rf /var/lib/apt/lists/*

# Install additional Python dependencies
COPY ./requirements.txt /opt/
RUN python3.9 -m pip install --no-cache-dir -r /opt/requirements.txt


# Now copy the rest of the src code into the image
COPY src /opt/src
COPY ./entry_point.sh ./fix_line_endings.sh /opt/

# Set permissions and run fix_line_endings script
RUN chmod +x /opt/entry_point.sh /opt/fix_line_endings.sh && \
    /opt/fix_line_endings.sh "/opt/src" && \
    /opt/fix_line_endings.sh "/opt/entry_point.sh"

# Set working directory and environment variables
WORKDIR /opt/src
# Download the intended model - we are caching the model in the image
RUN python /opt/src/prediction/download_model.py

ENV PYTHONUNBUFFERED=TRUE \
    PYTHONDONTWRITEBYTECODE=TRUE \
    PATH="/opt/src:${PATH}" \
    PYTHONPATH="/opt/src:${PYTHONPATH}" \
    TORCH_HOME="/opt" \
    MPLCONFIGDIR="/opt" \
    TRANSFORMERS_CACHE="/opt"

# Set permissions, non-root user, and entrypoint
RUN chown -R 1000:1000 /opt && chmod -R 777 /opt
USER 1000
ENTRYPOINT ["/opt/entry_point.sh"]
