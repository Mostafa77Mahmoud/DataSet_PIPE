import requests
import re
import logging
from typing import Tuple, Optional
from bs4 import BeautifulSoup
import time


class GoogleDocsLoader:
    """Class to handle Google Docs content extraction."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent':
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })

    def extract_doc_id(self, url: str) -> str:
        """Extract document ID from Google Docs URL."""
        pattern = r'/document/d/([a-zA-Z0-9-_]+)'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        raise ValueError(f"Could not extract document ID from URL: {url}")

    def download_document(self, url: str, max_retries: int = 3) -> str:
        """Download document content from Google Docs URL."""
        doc_id = self.extract_doc_id(url)
        export_url = f"https://docs.google.com/document/d/{doc_id}/export?format=html"

        self.logger.info(f"Downloading document from: {url}")
        self.logger.info(f"Export URL: {export_url}")

        for attempt in range(max_retries):
            try:
                response = self.session.get(export_url, timeout=300)
                response.raise_for_status()

                if response.content:
                    self.logger.info(
                        f"Successfully downloaded document (attempt {attempt + 1})"
                    )
                    return response.text
                else:
                    self.logger.warning(
                        f"Empty response received (attempt {attempt + 1})")

            except requests.RequestException as e:
                self.logger.warning(
                    f"Download attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = min(60, 5 * (2**attempt))  # Cap at 60 seconds
                    self.logger.info(
                        f"Waiting {wait_time} seconds before retry...")
                    time.sleep(wait_time)
                else:
                    raise Exception(
                        f"Failed to download document after {max_retries} attempts: {e}"
                    )

        raise Exception("Failed to download document")

    def extract_text_from_html(self, html_content: str) -> str:
        """Extract clean text from HTML content."""
        self.logger.info("Extracting text from HTML content")

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['style', 'script', 'meta', 'link']):
                element.decompose()

            # Get text content
            text = soup.get_text()

            # Clean up the text
            text = self._clean_extracted_text(text)

            self.logger.info(f"Extracted text length: {len(text)} characters")
            return text

        except Exception as e:
            self.logger.error(f"Failed to extract text from HTML: {e}")
            raise

    def _clean_extracted_text(self, text: str) -> str:
        """Clean extracted text by removing unwanted elements."""
        # Remove multiple whitespaces and normalize line breaks
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n\n', text)

        # Remove common unwanted patterns
        unwanted_patterns = [
            r'Table of Contents.*?(?=\n\n|\n[A-Z])',  # Remove TOC
            r'Page \d+',  # Remove page numbers
            r'^\s*\d+\s*$',  # Remove standalone numbers
            r'Appendix.*?(?=\n\n|\n[A-Z])',  # Remove appendix
            r'Bibliography.*?(?=\n\n|\n[A-Z])',  # Remove bibliography
            r'References.*?(?=\n\n|\n[A-Z])',  # Remove references
        ]

        for pattern in unwanted_patterns:
            text = re.sub(pattern,
                          '',
                          text,
                          flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)

        # Clean up Arabic punctuation and normalize
        text = self._normalize_arabic_text(text)

        # Remove extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

        return text

    def _normalize_arabic_text(self, text: str) -> str:
        """Normalize Arabic text and punctuation."""
        # Normalize Arabic punctuation
        arabic_punctuation_map = {
            '؟': '؟',  # Arabic question mark
            '؛': '؛',  # Arabic semicolon
            '،': '،',  # Arabic comma
        }

        for old, new in arabic_punctuation_map.items():
            text = text.replace(old, new)

        # Remove extra spaces around Arabic punctuation
        text = re.sub(r'\s+([؟؛،])', r'\1', text)
        text = re.sub(r'([؟؛،])\s+', r'\1 ', text)

        return text

    def load_documents(self, ar_url: str, en_url: str) -> Tuple[str, str]:
        """Load both Arabic and English documents."""
        self.logger.info("Starting document loading process")

        # Download Arabic document
        self.logger.info("Downloading Arabic document...")
        ar_html = self.download_document(ar_url)
        ar_text = self.extract_text_from_html(ar_html)

        # Download English document
        self.logger.info("Downloading English document...")
        en_html = self.download_document(en_url)
        en_text = self.extract_text_from_html(en_html)

        self.logger.info(f"Arabic document length: {len(ar_text)} characters")
        self.logger.info(f"English document length: {len(en_text)} characters")

        return ar_text, en_text
