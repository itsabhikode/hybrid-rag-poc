# Use Python 3.13 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gnupg2 \
    unixodbc \
    unixodbc-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*


#spacy dependencies


# Install Poetry
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VERSION=1.7.1
RUN curl -sSL https://install.python-poetry.org | POETRY_VERSION=${POETRY_VERSION} python3 -
ENV PATH="${POETRY_HOME}/bin:$PATH"



# Copy poetry files
COPY pyproject.toml ./
COPY poetry.lock* ./

# Configure poetry to not create a virtual environment
RUN poetry config virtualenvs.create false

    # Install dependencies (will create lock file if it doesn't exist)
    RUN poetry install --no-interaction --no-ansi
    
    # Fix numpy compatibility issue and reinstall spaCy
    RUN pip install --upgrade numpy
    RUN pip uninstall -y spacy thinc
    RUN pip install spacy==3.7.5
    RUN python -m spacy download en_core_web_sm

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p logs temp cache_dir


# Set environment variables
ENV PYTHONPATH=/
ENV PYTHONUNBUFFERED=1

# Change working directory to app folder
WORKDIR /app

# Expose port (if your app uses one)
EXPOSE 8000
 #entrypont script

CMD ["poetry", "run", "python", "-m", "hypercorn", "app:app", "--bind", "0.0.0.0:8000", "--reload"] 
