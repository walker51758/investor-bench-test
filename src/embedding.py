import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Union

import httpx
from loguru import logger
from pydantic import BaseModel


class EmbeddingObject(BaseModel):
    object: Literal["embedding"]
    embedding: List[float]
    index: int


class EmbeddingSuccessResponse(BaseModel):
    object: Literal["list"]
    data: List[EmbeddingObject]
    model: str
    usage: Dict[str, int]


class ErrorObject(BaseModel):
    message: str
    type: str
    param: Union[None, str]
    code: Union[None, str]


class EmbeddingErrorResponse(BaseModel):
    error: ErrorObject


class OpenAIEmbeddingError(Exception):
    def __init__(self, message: str, error_type: str) -> None:
        self.message = f"OpenAI Embedding failed, with error type {error_type}, error message: *[{message}]*"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class MiniMaxEmbeddingError(Exception):
    def __init__(self, message: str, error_type: str) -> None:
        self.message = f"MiniMax Embedding failed, with error type {error_type}, error message: *[{message}]*"
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class EmbeddingModel(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def __call__(self, texts: List[str]) -> List[List[float]]:
        pass


class OpenAIEmbedding(EmbeddingModel):
    def __init__(self, emb_config: Dict) -> None:
        self.config = emb_config
        logger.trace(f"EMB-Initializing OpenAIEmbedding with config: {self.config}")
        # auth
        try:
            openai_api_key = os.environ["OPENAI_API_KEY"]
        except KeyError as e:
            logger.error("Can not find openai api key")
            raise ValueError("Can not find openai api key") from e
        self.header = {
            "Authorization": f"Bearer {openai_api_key}",
            "Content-Type": "application/json",
        }

    def __call__(self, texts: Union[List[str], str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        with httpx.Client(timeout=self.config["embedding_timeout"]) as client:
            logger.trace(
                f"EMB-Calling OpenAIEmbedding with model: {self.config['emb_model_name']}, endpoint: {self.config['request_endpoint']}"
            )
            request_data = {
                "input": texts,
                "model": self.config["emb_model_name"],
                "encoding_format": "float",
            }

            response = client.post(
                url=self.config["request_endpoint"],
                headers=self.header,
                json=request_data,
            )

            try:
                results = EmbeddingSuccessResponse(**response.json())
                logger.trace("EMB-OpenAIEmbedding success response")
            except Exception as e:
                try:
                    error_response = EmbeddingErrorResponse(**response.json())
                    logger.error(
                        f"EMB-OpenAIEmbedding failed with error: {error_response.error.message}, error type: {error_response.error.type}"
                    )
                    raise OpenAIEmbeddingError(
                        message=error_response.error.message,
                        error_type=error_response.error.type,
                    ) from e
                except Exception:
                    response.raise_for_status()
                    logger.error("EMB-OpenAIEmbedding failed with unknown error")

            # ensure the order and return
            embeddings = sorted(results.data, key=lambda x: x.index)  # type: ignore
            return [i.embedding for i in embeddings]


class MiniMaxEmbedding(EmbeddingModel):
    def __init__(self, emb_config: Dict) -> None:
        self.config = emb_config
        logger.trace(f"EMB-Initializing MiniMaxEmbedding with config: {self.config}")
        # auth - MiniMax requires Bearer prefix for embedding API
        api_key_env = emb_config.get("api_key_env", "MINIMAX_EMB_API_KEY")
        try:
            api_key = os.environ[api_key_env]
        except KeyError as e:
            logger.error(f"Can not find {api_key_env} environment variable")
            raise ValueError(f"Can not find {api_key_env} environment variable") from e
        self.header = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def __call__(self, texts: Union[List[str], str]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]

        max_retries = 3
        for attempt in range(max_retries):
            try:
                return self._call_embedding(texts)
            except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.TimeoutException) as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"EMB-MiniMaxEmbedding connection error (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(
                        f"EMB-MiniMaxEmbedding failed after {max_retries} attempts: {e}"
                    )
                    raise MiniMaxEmbeddingError(
                        message=str(e),
                        error_type="connection_error",
                    ) from e

    def _call_embedding(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(timeout=self.config["embedding_timeout"]) as client:
            logger.trace(
                f"EMB-Calling MiniMaxEmbedding with model: {self.config['emb_model_name']}, endpoint: {self.config['request_endpoint']}"
            )
            request_data = {
                "texts": texts,
                "model": self.config["emb_model_name"],
                "type": "db",
            }

            response = client.post(
                url=self.config["request_endpoint"],
                headers=self.header,
                json=request_data,
            )

            try:
                result = response.json()
                logger.trace("EMB-MiniMaxEmbedding response received")
            except Exception as e:
                logger.error(f"EMB-MiniMaxEmbedding failed to parse JSON response: {e}")
                raise MiniMaxEmbeddingError(
                    message=str(e),
                    error_type="json_parse_error",
                ) from e

            if result.get("base_resp", {}).get("status_code", 0) != 0:
                err_msg = result.get("base_resp", {}).get("status_msg", "Unknown error")
                logger.error(f"EMB-MiniMaxEmbedding API error: {err_msg}")
                raise MiniMaxEmbeddingError(message=err_msg, error_type="api_error")

            vectors = result.get("vectors", [])
            if not vectors:
                logger.error("EMB-MiniMaxEmbedding returned no vectors")
                raise MiniMaxEmbeddingError(message="No vectors returned", error_type="empty_response")

            return vectors
