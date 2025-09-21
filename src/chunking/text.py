from __future__ import annotations

import re
from typing import List


def _clean_text(text: str) -> str:
    """Clean and normalize text for chunking"""
    # Normalize line endings
    text = re.sub(r"\r\n|\r", "\n", text)
    # Reduce multiple newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Advanced text chunking with sentence boundary preference
    
    Args:
        text: Input text to chunk
        chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
    
    Returns:
        List of text chunks
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return []
    
    if len(cleaned) <= chunk_size:
        return [cleaned]
    
    chunks: List[str] = []
    start = 0
    
    while start < len(cleaned):
        end = min(len(cleaned), start + chunk_size)
        
        # Try to end at a sentence boundary for better context
        period = cleaned.rfind(". ", start, end)
        exclamation = cleaned.rfind("! ", start, end)
        question = cleaned.rfind("? ", start, end)
        
        # Find the best sentence boundary
        sentence_end = max(period, exclamation, question)
        
        # Use sentence boundary if it's not too close to start
        if sentence_end != -1 and sentence_end > start + int(chunk_size * 0.6):
            end = sentence_end + 1
        
        # If no good sentence boundary, try paragraph boundary
        elif end < len(cleaned):
            paragraph = cleaned.rfind("\n\n", start, end)
            if paragraph != -1 and paragraph > start + int(chunk_size * 0.5):
                end = paragraph + 2
        
        part = cleaned[start:end].strip()
        if part:
            chunks.append(part)
        
        if end >= len(cleaned):
            break
        
        # Move start position with overlap
        start = max(0, end - overlap)
    
    return chunks


def chunk_news_article(article_text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Specialized chunking for news articles"""
    # For news articles, we might want to preserve structure
    cleaned = _clean_text(article_text)
    
    # Split by paragraphs first if they exist
    paragraphs = cleaned.split("\n\n")
    
    if len(paragraphs) == 1:
        # Single paragraph, use regular chunking
        return chunk_text(cleaned, chunk_size, overlap)
    
    chunks = []
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        
        # If adding this paragraph would exceed chunk size
        if len(current_chunk) + len(paragraph) + 2 > chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too long, chunk it
            if len(paragraph) > chunk_size:
                sub_chunks = chunk_text(paragraph, chunk_size, overlap)
                chunks.extend(sub_chunks[:-1])  # Add all but last
                current_chunk = sub_chunks[-1] if sub_chunks else ""
            else:
                current_chunk = paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks


def chunk_company_context(company_profile, chunk_size: int = 800) -> List[str]:
    """Create chunks from company profile for context"""
    chunks = []
    
    # Basic info chunk
    basic_info = f"""
    Company: {company_profile.company_name}
    Industry: {company_profile.industry}
    Main Products/Services: {', '.join(company_profile.main_products_services)}
    Target Markets: {', '.join(company_profile.target_markets)}
    """
    chunks.append(basic_info.strip())
    
    # Business details chunk
    if company_profile.business_model or company_profile.revenue_streams:
        business_info = f"""
        Business Model: {company_profile.business_model or 'Not specified'}
        Revenue Streams: {', '.join(company_profile.revenue_streams) if company_profile.revenue_streams else 'Not specified'}
        """
        chunks.append(business_info.strip())
    
    # Financial context chunk
    if any([company_profile.annual_revenue, company_profile.employee_count, company_profile.market_share]):
        financial_info = f"""
        Annual Revenue: ${company_profile.annual_revenue:,.0f} USD" if company_profile.annual_revenue else "Not specified"
        Employee Count: {company_profile.employee_count:,}" if company_profile.employee_count else "Not specified"
        Market Share: {company_profile.market_share}%" if company_profile.market_share else "Not specified"
        """
        chunks.append(financial_info.strip())
    
    return chunks

