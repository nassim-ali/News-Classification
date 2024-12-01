# Use a slim Debian base image
FROM debian:bullseye-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV SPARK_VERSION=3.5.3
ENV HADOOP_VERSION=3.3.6
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install necessary tools and OpenJDK
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    curl \
    bash \
    gnupg \
    openjdk-11-jdk \
    python3 \
    python3-pip \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Hadoop
RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-${HADOOP_VERSION}/hadoop-${HADOOP_VERSION}.tar.gz && \
    tar -xvzf hadoop-${HADOOP_VERSION}.tar.gz && \
    mv hadoop-${HADOOP_VERSION} /usr/local/hadoop && \
    rm hadoop-${HADOOP_VERSION}.tar.gz

# Set HADOOP_HOME and update PATH
ENV HADOOP_HOME=/usr/local/hadoop
ENV PATH="$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin"

# Install Spark
RUN wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    tar -xvzf spark-${SPARK_VERSION}-bin-hadoop3.tgz && \
    mv spark-${SPARK_VERSION}-bin-hadoop3 /usr/local/spark && \
    rm spark-${SPARK_VERSION}-bin-hadoop3.tgz

# Set SPARK_HOME and update PATH
ENV SPARK_HOME=/usr/local/spark
ENV PATH="$PATH:$SPARK_HOME/bin"

# Install Python dependencies
WORKDIR /app
COPY requirement.txt /app/
RUN pip3 install --no-cache-dir -r requirement.txt

# Copy application files
COPY . /app/
