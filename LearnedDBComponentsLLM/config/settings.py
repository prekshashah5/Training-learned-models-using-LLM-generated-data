"""
settings.py
Shared settings and constants for the project.
"""

import os

# Default materialized sample count for MSCN bitmap generation
NUM_MATERIALIZED_SAMPLES = int(os.getenv("NUM_MATERIALIZED_SAMPLES", 1000))

# Default training parameters
DEFAULT_BATCH_SIZE = int(os.getenv("BATCH_SIZE", 1024))
DEFAULT_HIDDEN_UNITS = int(os.getenv("HIDDEN_UNITS", 256))
DEFAULT_LEARNING_RATE = float(os.getenv("LEARNING_RATE", 0.0001))

# Active Learning defaults
DEFAULT_AL_ROUNDS = int(os.getenv("AL_ROUNDS", 5))
DEFAULT_AL_ACQUIRE = int(os.getenv("AL_ACQUIRE", 200))
DEFAULT_AL_EPOCHS = int(os.getenv("AL_EPOCHS", 10))

# LLM Generation defaults
DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_MODEL_NAME = os.getenv("MODELS", "llama3.2")
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 0.8))
DEFAULT_GEN_BATCH_SIZE = int(os.getenv("GEN_BATCH_SIZE", 20))
DEFAULT_TOTAL_QUERIES = int(os.getenv("TOTAL_QUERIES", 100))

# Database timeouts
DB_TIMEOUT = int(os.getenv("DB_TIMEOUT", 6000))
MAX_DB_RETRIES = int(os.getenv("MAX_DB_RETRIES", 2))
