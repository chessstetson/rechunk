"""
Extracted content models and :class:`ExtractedContentService` protocol (v12).

Phase A: contracts only. No default service implementation yet.
"""

from rechunk.extracted_content.filesystem import FilesystemExtractedContentService
from rechunk.extracted_content.models import ExtractedContent, SourceDocumentRef
from rechunk.extracted_content.protocol import ExtractedContentService

__all__ = [
    "ExtractedContent",
    "ExtractedContentService",
    "FilesystemExtractedContentService",
    "SourceDocumentRef",
]
