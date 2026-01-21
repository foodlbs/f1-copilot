"""
Configuration Management

Central configuration for the F1 Knowledge Base system.
Load settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Config:
    """Application configuration"""

    # API Keys
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")

    # Pinecone Configuration
    PINECONE_ENVIRONMENT: str = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "f1-race-knowledge")

    # Data Collection Settings
    START_YEAR: int = int(os.getenv("START_YEAR", "2017"))
    DATA_CACHE_DIR: str = os.getenv("DATA_CACHE_DIR", "./cache/f1_data")
    FASTF1_CACHE_DIR: str = os.getenv("FASTF1_CACHE_DIR", "./cache/fastf1")

    # Vector Database Settings
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))
    BATCH_SIZE: int = int(os.getenv("BATCH_SIZE", "100"))

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_DIR: str = os.getenv("LOG_DIR", "./logs")

    # Data Export
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    @classmethod
    def validate(cls) -> bool:
        """
        Validate configuration

        Returns:
            True if configuration is valid
        """
        errors = []

        if not cls.PINECONE_API_KEY:
            errors.append("PINECONE_API_KEY not set")

        if not cls.OPENAI_API_KEY:
            errors.append("OPENAI_API_KEY not set")

        if cls.START_YEAR < 1950 or cls.START_YEAR > 2030:
            errors.append(f"Invalid START_YEAR: {cls.START_YEAR}")

        if errors:
            print("❌ Configuration errors:")
            for error in errors:
                print(f"   • {error}")
            return False

        return True

    @classmethod
    def create_directories(cls):
        """Create necessary directories"""
        dirs = [
            cls.DATA_CACHE_DIR,
            cls.FASTF1_CACHE_DIR,
            cls.LOG_DIR,
            cls.DATA_DIR
        ]

        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        """Print current configuration (safe - no API keys)"""
        print("\n" + "="*60)
        print("CONFIGURATION")
        print("="*60)
        print(f"Pinecone Index: {cls.PINECONE_INDEX_NAME}")
        print(f"Pinecone Region: {cls.PINECONE_ENVIRONMENT}")
        print(f"Embedding Model: {cls.EMBEDDING_MODEL}")
        print(f"Embedding Dimension: {cls.EMBEDDING_DIMENSION}")
        print(f"Data Start Year: {cls.START_YEAR}")
        print(f"Data Cache: {cls.DATA_CACHE_DIR}")
        print(f"FastF1 Cache: {cls.FASTF1_CACHE_DIR}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Log Level: {cls.LOG_LEVEL}")
        print("="*60 + "\n")


# Singleton instance
config = Config()


if __name__ == "__main__":
    # Test configuration
    config.print_config()

    if config.validate():
        print("✅ Configuration is valid")
        config.create_directories()
        print("✅ Directories created")
    else:
        print("❌ Configuration is invalid")
