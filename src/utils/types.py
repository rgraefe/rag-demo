from typing import List, Any, Dict

from pydantic import BaseModel
from fastapi_poe.types import (
    QueryRequest,
    ReportErrorRequest,
    ReportFeedbackRequest,
    SettingsRequest,
)


class Document(BaseModel):
    doc_id: str
    text: str


class AddDocumentsRequest(BaseModel):
    """Request parameters for an add_documents request."""

    documents: List[Document]


class LLMQueryRequest(QueryRequest):
    params: Dict[str, Any]