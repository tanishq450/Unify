"""
Table Extraction for Financial PDFs
Production-ready pipeline for extracting tables from uploaded documents

Why this matters:
- 40-60% of financial data lives in tables
- Text-only RAG misses revenue breakdowns, segment data, comparisons
- Tables = the difference between generic and accurate answers

Pipeline:
1. Detect tables in PDF
2. Extract to structured format (Markdown/TEDS)
3. Store with bounding box for citation
4. Retrieve alongside text chunks
"""

import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class TableData:
    """Extracted table with metadata"""
    table_id: str
    page: int
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    markdown: str  # Markdown representation
    teds: Optional[str] = None  # TEDS format for complex tables
    confidence: float = 1.0
    headers: List[str] = None
    num_rows: int = 0
    num_cols: int = 0
    table_type: str = "simple"  # simple, merged, hierarchical


@dataclass
class ExtractedContent:
    """Any extracted content from PDF"""
    content_type: str  # "text", "table", "chart", "image"
    content: str
    page: int
    bbox: Optional[Tuple[float, float, float, float]] = None
    metadata: Dict = None


# ============================================================================
# METHOD 1: LlamaParse (Best for production - 90%+ table accuracy)
# ============================================================================

class LlamaParseExtractor:
    """
    LlamaParse for table extraction

    Pros:
    - 90.7% accuracy on tables (ParseBench)
    - Handles merged cells, hierarchical headers
    - Returns Markdown directly
    - Preserves numerical formatting

    Cons:
    - Paid API ($0.003/page at time of writing)
    - Requires internet

    Setup:
    1. pip install llama-parse
    2. Get API key: https://cloud.llamaindex.ai
    3. Set LLAMA_CLOUD_API_KEY env var
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("LLAMA_API_KEY") or os.getenv("LLAMA_CLOUD_API_KEY")

        if not self.api_key:
            raise ValueError(
                "LlamaParse API key required. "
                "Set LLAMA_API_KEY or LLAMA_CLOUD_API_KEY in your .env file. "
                "Get a free key at: https://cloud.llamaindex.ai"
            )

        self.parser = None  # Lazy load

    def _init_parser(self):
        """Lazy initialize parser"""
        from llama_parse import LlamaParse

        self.parser = LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            verbose=True,
            do_not_cache=True,
        )

    def extract_from_pdf(self, pdf_path: str) -> List[TableData]:
        """
        Extract all tables from PDF

        Returns:
            List of TableData objects with markdown content
        """

        if not self.parser:
            self._init_parser()

        # Parse entire document
        documents = self.parser.load_data(pdf_path)

        # Parse response contains markdown with table markers
        tables = []
        
        for doc_idx, doc in enumerate(documents):
            full_text = doc.text
            # Split by table markers in markdown
            table_blocks = self._extract_table_blocks(full_text)

            for i, block in enumerate(table_blocks):
                table = TableData(
                    table_id=f"doc_{doc_idx}_table_{i}",
                    page=doc.metadata.get("page_number", self._infer_page(block, full_text)),
                    bbox=(0, 0, 0, 0),  # LlamaParse doesn't provide bbox
                    markdown=block,
                    headers=self._extract_headers(block),
                    num_rows=self._count_rows(block),
                    num_cols=self._count_cols(block),
                    table_type=self._classify_table(block),
                    confidence=0.9  # LlamaParse is reliable
                )
                tables.append(table)

        return tables

    def _extract_table_blocks(self, markdown_text: str) -> List[str]:
        """Extract table blocks from markdown"""

        lines = markdown_text.split('\n')
        table_blocks = []
        in_table = False
        current_table = []

        for line in lines:
            # Detect table start (header with | separators)
            if '|' in line and not in_table:
                in_table = True
                current_table = [line]
            elif in_table:
                if '|' in line or line.strip() == '':
                    current_table.append(line)
                else:
                    # Table ended
                    if current_table:
                        table_blocks.append('\n'.join(current_table))
                    in_table = False
                    current_table = []

        # Don't forget last table
        if current_table and len(current_table) > 2:
            table_blocks.append('\n'.join(current_table))

        return table_blocks

    def _extract_headers(self, markdown_table: str) -> List[str]:
        """Extract header row from markdown table"""
        lines = markdown_table.strip().split('\n')
        if len(lines) < 2:
            return []

        # First line is header
        header_line = lines[0]
        headers = [h.strip() for h in header_line.split('|')]

        # Clean empty strings from edges
        if headers and headers[0] == '':
            headers = headers[1:]
        if headers and headers[-1] == '':
            headers = headers[:-1]

        return headers

    def _count_rows(self, markdown_table: str) -> int:
        """Count data rows (excluding header and separator)"""
        lines = markdown_table.strip().split('\n')

        # Subtract header and separator line
        data_rows = len(lines) - 2

        return max(0, data_rows)

    def _count_cols(self, markdown_table: str) -> int:
        """Count columns"""
        lines = markdown_table.strip().split('\n')
        if not lines:
            return 0

        header_line = lines[0]
        cols = len([c for c in header_line.split('|') if c.strip()])

        return cols

    def _classify_table(self, markdown_table: str) -> str:
        """Classify table complexity"""

        # Check for merged cells indicators
        if '[merged]' in markdown_table.lower():
            return "merged"

        # Check for hierarchical headers (multiple header rows)
        lines = markdown_table.strip().split('\n')
        if len(lines) > 3:
            # Check if second row looks like a separator or data
            if '---' not in lines[1]:
                return "hierarchical"

        return "simple"

    def _infer_page(self, table_block: str, full_text: str) -> int:
        """Infer page number from position in document"""

        # Rough estimation based on character position
        # For accurate page numbers, use pdfplumber alongside
        char_position = full_text.find(table_block)
        if char_position < 0:
            return 1

        # Assume ~3000 chars per page
        estimated_page = (char_position // 3000) + 1

        return min(estimated_page, 100)  # Cap at reasonable max


# ============================================================================
# METHOD 2: pdfplumber (Free, local, good for simple tables)
# ============================================================================

class PdfplumberExtractor:
    """
    pdfplumber for local table extraction

    Pros:
    - Free, open source
    - Runs locally (no API)
    - Provides bounding boxes
    - Good for well-structured tables

    Cons:
    - Struggles with merged cells
    - ~70-80% accuracy on complex tables
    - Requires PDF to have embedded text (not scanned)

    Setup:
    pip install pdfplumber pandas
    """

    def __init__(self):
        try:
            import pdfplumber
            import pandas as pd
        except ImportError:
            raise ImportError(
                "Install pdfplumber: pip install pdfplumber pandas"
            )

        self.pdfplumber = pdfplumber
        self.pd = pd

    def extract_from_pdf(self, pdf_path: str) -> List[TableData]:
        """Extract tables with bounding boxes"""

        tables = []
        table_id = 0

        with self.pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                # Extract tables from page
                page_tables = page.extract_tables()

                for table_idx, table_data in enumerate(page_tables):
                    if not table_data or len(table_data) < 2:
                        continue

                    # Convert to markdown
                    markdown = self._table_to_markdown(table_data)

                    # Get bounding box (approximate)
                    bbox = self._get_table_bbox(page, table_data)

                    table = TableData(
                        table_id=f"table_{table_id}",
                        page=page_num,
                        bbox=bbox,
                        markdown=markdown,
                        headers=self._extract_headers_pd(table_data),
                        num_rows=len(table_data) - 1,  # Exclude header
                        num_cols=len(table_data[0]) if table_data[0] else 0,
                        table_type="simple",  # pdfplumber handles simple tables
                        confidence=0.75  # Lower confidence than LlamaParse
                    )

                    tables.append(table)
                    table_id += 1

        return tables

    def _table_to_markdown(self, table_data: List[List[str]]) -> str:
        """Convert 2D array to markdown table"""

        if not table_data or not table_data[0]:
            return ""

        lines = []

        # Header row
        header = table_data[0]
        header_cleaned = [str(cell).strip() if cell else '' for cell in header]
        lines.append('| ' + ' | '.join(header_cleaned) + ' |')

        # Separator row
        lines.append('| ' + ' | '.join(['---'] * len(header)) + ' |')

        # Data rows
        for row in table_data[1:]:
            if not row:
                continue
            row_cleaned = [str(cell).strip() if cell else '' for cell in row]
            # Pad row to match header length
            while len(row_cleaned) < len(header_cleaned):
                row_cleaned.append('')
            lines.append('| ' + ' | '.join(row_cleaned[:len(header_cleaned)]) + ' |')

        return '\n'.join(lines)

    def _get_table_bbox(
        self,
        page,
        table_data: List[List[str]]
    ) -> Tuple[float, float, float, float]:
        """Get approximate bounding box for table"""

        # pdfplumber doesn't directly give table bbox
        # We estimate from page layout

        # Get all horizontal and vertical lines
        h_lines = page.chars  # Fallback to full page

        # Simple approximation: assume table spans most of page width
        page_bbox = page.bbox
        page_width = page_bbox[2] - page_bbox[0]
        page_height = page_bbox[3] - page_bbox[1]

        # Estimate based on table size
        num_rows = len(table_data)
        row_height = page_height / 50  # Assume ~50 rows per page

        return (
            page_bbox[0],  # x1 (left)
            page_bbox[1],  # y1 (top) - would need better estimation
            page_bbox[2],  # x2 (right)
            page_bbox[1] + (num_rows * row_height)  # y2 (bottom)
        )

    def _extract_headers_pd(self, table_data: List[List[str]]) -> List[str]:
        """Extract headers from table data"""

        if not table_data or not table_data[0]:
            return []

        return [str(h).strip() if h else '' for h in table_data[0]]


# ============================================================================
# METHOD 3: GPT-4V / Claude (Best for complex/irregular tables)
# ============================================================================

class VisionModelExtractor:
    """
    Vision model (GPT-4V, Claude, Qwen-VL) for table extraction

    Pros:
    - Handles complex layouts
    - Can interpret merged cells
    - Good for scanned PDFs (OCR + extraction)

    Cons:
    - Expensive (~$0.01-0.03 per page)
    - Slower (3-8 seconds per table)
    - Requires API key

    Setup:
    pip install openai pillow
    export OPENAI_API_KEY=...
    """

    def __init__(self, model: str = "gpt-4o", api_key: Optional[str] = None):
        self.model = model

        if model.startswith("gpt"):
            from Model_loader.llm import ModelLoader
            loader = ModelLoader()
            loader.load_models()
            self.client = loader.client
        elif model.startswith("claude"):
            self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
            if not self.api_key:
                raise ValueError("ANTHROPIC_API_KEY required")
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
        else:
            raise ValueError(f"Unsupported model: {model}")

        try:
            from PIL import Image
            self.Image = Image
        except ImportError:
            raise ImportError("Install pillow: pip install pillow")

    def extract_from_pdf(
        self,
        pdf_path: str,
        table_pages: List[int] = None
    ) -> List[TableData]:
        """
        Extract tables from specific pages

        Args:
            pdf_path: Path to PDF
            table_pages: List of page numbers (1-indexed) to process
                        If None, processes all pages
        """

        # Convert PDF pages to images
        page_images = self._pdf_to_images(pdf_path, table_pages)

        tables = []
        for page_num, img in page_images:
            # Extract tables from page image
            table_data = self._extract_from_image(img, page_num)
            tables.extend(table_data)

        return tables

    def _pdf_to_images(
        self,
        pdf_path: str,
        table_pages: List[int] = None
    ) -> List[Tuple[int, any]]:
        """Convert PDF pages to images"""

        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("Install PyMuPDF: pip install pymupdf")

        doc = fitz.open(pdf_path)
        page_images = []

        pages_to_process = table_pages or list(range(1, len(doc) + 1))

        for page_num in pages_to_process:
            if page_num < 1 or page_num > len(doc):
                continue

            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution

            # Convert to PIL Image
            img = self.Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            page_images.append((page_num, img))

        doc.close()
        return page_images

    def _extract_from_image(self, img, page_num: int) -> List[TableData]:
        """Extract tables from page image using vision model"""

        # First, detect if there are tables
        detection_prompt = """
        Does this page contain any tables?
        Reply with ONLY "YES" or "NO".
        """

        # For simplicity, assume all pages have tables
        # In production, add a detection step

        # Extract tables
        extraction_prompt = """
        Extract ALL tables from this page into Markdown format.

        Requirements:
        1. Preserve exact numerical values (don't round!)
        2. Include all headers and subheaders
        3. For merged cells, use [merged] placeholder
        4. Maintain column alignment
        5. Include units in headers (e.g., "$ in billions")

        If multiple tables, separate with "---TABLE BREAK---"

        Output format:
        ```markdown
        | Header 1 | Header 2 |
        |----------|----------|
        | Value 1  | Value 2  |
        ```

        If no tables found, reply "NO TABLES FOUND".
        """

        # Call vision model
        if self.model.startswith("gpt"):
            response = self._call_gpt4v(img, extraction_prompt)
        else:
            response = self._call_claude(img, extraction_prompt)

        # Parse response
        tables = self._parse_vision_response(response, page_num)

        return tables

    def _call_gpt4v(self, img, prompt: str) -> str:
        """Call GPT-4V"""

        import base64
        from io import BytesIO

        # Convert image to base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{img_base64}"
                    }}
                ]
            }],
            max_tokens=2000
        )

        return response.choices[0].message.content

    def _call_claude(self, img, prompt: str) -> str:
        """Call Claude"""

        import base64
        from io import BytesIO

        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        response = self.client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img_base64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }]
        )

        return response.content[0].text

    def _parse_vision_response(self, response: str, page_num: int) -> List[TableData]:
        """Parse vision model response into TableData objects"""

        tables = []

        # Split by table breaks
        table_blocks = response.split("---TABLE BREAK---")

        for i, block in enumerate(table_blocks):
            block = block.strip()

            if "NO TABLES FOUND" in block:
                continue

            # Extract markdown table
            if "```markdown" in block:
                # Extract from code block
                start = block.find("```markdown") + 11
                end = block.find("```", start)
                markdown = block[start:end].strip()
            else:
                markdown = block

            if not markdown or '|' not in markdown:
                continue

            table = TableData(
                table_id=f"table_vision_{i}",
                page=page_num,
                bbox=(0, 0, 0, 0),  # Vision models don't provide bbox
                markdown=markdown,
                headers=self._extract_headers_md(markdown),
                num_rows=self._count_rows_md(markdown),
                num_cols=self._count_cols_md(markdown),
                table_type="complex",
                confidence=0.85
            )

            tables.append(table)

        return tables

    def _extract_headers_md(self, markdown: str) -> List[str]:
        """Extract headers from markdown table"""
        lines = markdown.strip().split('\n')
        if len(lines) < 2:
            return []

        header_line = lines[0]
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        return headers

    def _count_rows_md(self, markdown: str) -> int:
        """Count data rows"""
        lines = markdown.strip().split('\n')
        # Subtract header and separator
        return max(0, len(lines) - 2)

    def _count_cols_md(self, markdown: str) -> int:
        """Count columns"""
        lines = markdown.strip().split('\n')
        if not lines:
            return 0

        header_line = lines[0]
        return len([c for c in header_line.split('|') if c.strip()])


# ============================================================================
# UNIFIED EXTRACTOR - Auto-selects best method
# ============================================================================

class UnifiedTableExtractor:
    """
    Unified table extractor with automatic method selection

    Strategy:
    1. Try LlamaParse (best accuracy) if API key available
    2. Fall back to pdfplumber (free, local)
    3. For complex tables, use vision model

    Usage:
        extractor = UnifiedTableExtractor(
            llama_parse_key="...",
            openai_key="..."
        )
        tables = extractor.extract("document.pdf")
    """

    def __init__(
        self,
        llama_parse_key: Optional[str] = None,
        openai_key: Optional[str] = None,
        anthropic_key: Optional[str] = None,
        prefer_local: bool = False
    ):
        """
        Initialize extractor

        Args:
            llama_parse_key: LlamaParse API key (recommended)
            openai_key: OpenAI API key (for fallback)
            anthropic_key: Anthropic API key (alternative fallback)
            prefer_local: If True, use pdfplumber even if API keys available
        """

        self.llama_parse_key = llama_parse_key or os.getenv("LLAMA_API_KEY") or os.getenv("LLAMA_CLOUD_API_KEY")
        self.openai_key = openai_key or os.getenv("OPENAI_API_KEY") or os.getenv("MESH_API_KEY")
        self.anthropic_key = anthropic_key or os.getenv("ANTHROPIC_API_KEY")
        self.prefer_local = prefer_local

        # Initialize extractors lazily
        self._llama_extractor = None
        self._pdfplumber_extractor = None
        self._vision_extractor = None

    def extract(self, pdf_path: str) -> List[TableData]:
        """
        Extract tables from PDF

        Automatically selects best method based on available tools
        """

        # Check file exists
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        # Select method
        if self.prefer_local:
            return self._extract_local(pdf_path)
        elif self.llama_parse_key:
            return self._extract_llama_parse(pdf_path)
        elif self.openai_key or self.anthropic_key:
            return self._extract_vision(pdf_path)
        else:
            return self._extract_local(pdf_path)

    def _extract_llama_parse(self, pdf_path: str) -> List[TableData]:
        """Extract using LlamaParse (best quality)"""

        if not self._llama_extractor:
            self._llama_extractor = LlamaParseExtractor(api_key=self.llama_parse_key)

        return self._llama_extractor.extract_from_pdf(pdf_path)

    def _extract_local(self, pdf_path: str) -> List[TableData]:
        """Extract using pdfplumber (free, local)"""

        if not self._pdfplumber_extractor:
            self._pdfplumber_extractor = PdfplumberExtractor()

        return self._pdfplumber_extractor.extract_from_pdf(pdf_path)

    def _extract_vision(self, pdf_path: str) -> List[TableData]:
        """Extract using vision model (best for complex tables)"""

        if not self._vision_extractor:
            model = "gpt-4o" if self.openai_key else "claude-3-opus-20240229"
            api_key = self.openai_key or self.anthropic_key
            self._vision_extractor = VisionModelExtractor(
                model=model,
                api_key=api_key
            )

        return self._vision_extractor.extract_from_pdf(pdf_path)

    def extract_with_fallback(self, pdf_path: str) -> List[TableData]:
        """
        Extract with cascading fallback

        Tries LlamaParse → pdfplumber → Vision
        """

        # Try LlamaParse first
        if self.llama_parse_key:
            try:
                print("Trying LlamaParse...")
                tables = self._extract_llama_parse(pdf_path)
                if tables:
                    print(f"LlamaParse found {len(tables)} tables")
                    return tables
            except Exception as e:
                print(f"LlamaParse failed: {e}")

        # Try pdfplumber
        try:
            print("Trying pdfplumber...")
            tables = self._extract_local(pdf_path)
            if tables:
                print(f"pdfplumber found {len(tables)} tables")
                return tables
        except Exception as e:
            print(f"pdfplumber failed: {e}")

        # Try vision model
        if self.openai_key or self.anthropic_key:
            try:
                print("Trying vision model...")
                tables = self._extract_vision(pdf_path)
                if tables:
                    print(f"Vision model found {len(tables)} tables")
                    return tables
            except Exception as e:
                print(f"Vision model failed: {e}")

        print("No tables found or all methods failed")
        return []


# ============================================================================
# Integration with RAG Pipeline
# ============================================================================

class TableAwareRAG:
    """
    RAG pipeline that includes tables

    Stores tables separately and retrieves them alongside text chunks
    """

    def __init__(self, table_extractor: UnifiedTableExtractor):
        self.table_extractor = table_extractor
        self.tables = []  # List[TableData]
        self.text_chunks = []  # List of text chunks

    def ingest(self, pdf_path: str, text_chunks: List[str] = None):
        """
        Ingest PDF with tables

        Args:
            pdf_path: Path to PDF
            text_chunks: Optional pre-extracted text chunks
        """

        print(f"Ingesting {pdf_path}...")

        # Extract tables
        print("Extracting tables...")
        self.tables = self.table_extractor.extract_with_fallback(pdf_path)
        print(f"Found {len(self.tables)} tables")

        # Store text chunks (if provided)
        if text_chunks:
            self.text_chunks = text_chunks

        # Create unified index
        self._build_index()

    def _build_index(self):
        """Build unified index of text + tables"""

        # In production, this would:
        # 1. Embed table markdown
        # 2. Store in Neo4j/VectorDB with metadata
        # 3. Link tables to surrounding text context

        self.index = {
            "text_chunks": self.text_chunks,
            "tables": [
                {
                    "id": t.table_id,
                    "page": t.page,
                    "markdown": t.markdown,
                    "headers": t.headers,
                    "type": t.table_type,
                    "confidence": t.confidence
                }
                for t in self.tables
            ]
        }

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        """
        Retrieve relevant content including tables

        Boosts table results for numerical queries
        """

        # Classify query
        is_numerical = self._is_numerical_query(query)

        results = []

        # For numerical queries, prioritize tables
        if is_numerical:
            # Simple keyword match for tables
            for table in self.index["tables"]:
                score = self._score_table(query, table)
                if score > 0.3:
                    results.append({
                        "type": "table",
                        "content": table["markdown"],
                        "score": score,
                        "metadata": {
                            "page": table["page"],
                            "table_id": table["id"]
                        }
                    })

        # Add text results (would use vector search in production)
        for i, chunk in enumerate(self.text_chunks):
            if query.lower() in chunk.lower():
                results.append({
                    "type": "text",
                    "content": chunk[:500],
                    "score": 0.5,
                    "metadata": {"chunk_id": i}
                })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:k]

    def _is_numerical_query(self, query: str) -> bool:
        """Check if query is asking for numerical data"""

        numerical_keywords = [
            "revenue", "profit", "margin", "growth", "increase", "decrease",
            "compare", "vs", "breakdown", "by segment", "by product",
            "table", "figure", "show me the numbers", "how much", "what %"
        ]

        query_lower = query.lower()
        return any(kw in query_lower for kw in numerical_keywords)

    def _score_table(self, query: str, table: Dict) -> float:
        """Simple scoring for table relevance"""

        score = 0.0

        # Header match
        for header in table.get("headers", []):
            if header.lower() in query.lower():
                score += 0.3

        # Markdown content match
        if query.lower() in table["markdown"].lower():
            score += 0.5

        # Numerical query boost
        if self._is_numerical_query(query):
            score += 0.2

        return score


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Example 1: Simple extraction with pdfplumber (free)
    print("=" * 60)
    print("Example 1: Free local extraction with pdfplumber")
    print("=" * 60)

    extractor = UnifiedTableExtractor(prefer_local=True)

    # Create a sample PDF for testing (in practice, use real PDF)
    # tables = extractor.extract("sample_financial.pdf")

    # Example 2: Production extraction with LlamaParse
    print("\n" + "=" * 60)
    print("Example 2: Production extraction with LlamaParse")
    print("=" * 60)

    # extractor_prod = UnifiedTableExtractor(
    #     llama_parse_key="your-api-key",
    #     openai_key="your-openai-key"  # For fallback
    # )
    # tables = extractor_prod.extract("earnings_report.pdf")

    # for table in tables:
    #     print(f"\nTable {table.table_id} (Page {table.page})")
    #     print(f"Headers: {table.headers}")
    #     print(f"Size: {table.num_rows} rows x {table.num_cols} cols")
    #     print(f"Type: {table.table_type}")
    #     print(f"Confidence: {table.confidence}")
    #     print(f"Markdown:\n{table.markdown[:500]}...")

    # Example 3: RAG with tables
    print("\n" + "=" * 60)
    print("Example 3: Table-aware RAG")
    print("=" * 60)

    # Sample text chunks (in practice, extract from PDF)
    text_chunks = [
        "Apple reported strong financial results for fiscal year 2024.",
        "iPhone revenue grew 5% year-over-year driven by iPhone 15 sales.",
        "Services revenue reached an all-time high of $85 billion."
    ]

    # rag = TableAwareRAG(extractor)
    # rag.ingest("sample.pdf", text_chunks=text_chunks)

    # query = "What was iPhone revenue growth?"
    # results = rag.retrieve(query, k=3)

    # print(f"\nQuery: {query}")
    # for r in results:
    #     print(f"\n[{r['type']}] Score: {r['score']:.2f}")
    #     print(f"Content: {r['content'][:200]}...")

    print("\nTable extraction ready for integration!")
