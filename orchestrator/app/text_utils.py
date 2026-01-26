"""
Text utilities for TTS preprocessing.

This module provides text sanitization to prepare LLM output for TTS synthesis,
preventing gibberish/fantasy language output from both Chatterbox and Coqui engines.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


def sanitize_for_tts(text: str) -> str:
    """Prepare LLM output for TTS synthesis.

    Removes markdown formatting, special characters, and normalizes text
    to prevent TTS engines from producing gibberish or fantasy language.

    Args:
        text: Raw text from LLM output

    Returns:
        Sanitized text suitable for TTS synthesis
    """
    if not text:
        return text

    original_text = text

    # 1. Unicode normalization (NFC for German umlauts: ä, ö, ü, ß)
    text = unicodedata.normalize('NFC', text)

    # 2. Remove markdown formatting
    text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
    text = re.sub(r'\*(.+?)\*', r'\1', text)       # *italic*
    text = re.sub(r'_(.+?)_', r'\1', text)         # _underline_
    text = re.sub(r'`(.+?)`', r'\1', text)         # `code`
    text = re.sub(r'#{1,6}\s*', '', text)          # # headers
    text = re.sub(r'~~(.+?)~~', r'\1', text)       # ~~strikethrough~~

    # 3. Remove RAG source citations and links
    text = re.sub(r'\[Quelle \d+:.*?\]', '', text)  # RAG sources [Quelle 1: ...]
    text = re.sub(r'\[Quelle:.*?\]', '', text)      # [Quelle: ...]
    text = re.sub(r'\[\d+\]', '', text)             # Numeric citations [1]
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)      # [link](url)
    text = re.sub(r'\[.*?\]', '', text)             # [any remaining brackets]

    # 4. Remove parenthetical content that may confuse TTS
    # Only remove if content looks like metadata/technical info
    text = re.sub(r'\(siehe .*?\)', '', text)       # (siehe Anhang)
    text = re.sub(r'\(vgl\..*?\)', '', text)        # (vgl. ...)
    text = re.sub(r'\(Stand:.*?\)', '', text)       # (Stand: 2024)

    # 5. Replace special characters with spoken equivalents
    replacements = {
        '€': ' Euro',
        '$': ' Dollar',
        '£': ' Pfund',
        '¥': ' Yen',
        '%': ' Prozent',
        '&': ' und',
        '+': ' plus',
        '=': ' gleich',
        '@': ' at',
        '#': '',
        '→': '',
        '←': '',
        '↔': '',
        '•': '',
        '·': '',
        '…': '...',
        '–': '-',  # En-dash to hyphen
        '—': '-',  # Em-dash to hyphen
        '"': '"',  # Smart quotes to normal
        '"': '"',
        ''': "'",
        ''': "'",
        '„': '"',
        '‚': "'",
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)

    # 6. Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # 7. Remove email addresses
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # 8. Expand common German abbreviations for natural speech
    abbreviations = {
        'z.B.': 'zum Beispiel',
        'z. B.': 'zum Beispiel',
        'd.h.': 'das heißt',
        'd. h.': 'das heißt',
        'u.a.': 'unter anderem',
        'u. a.': 'unter anderem',
        'usw.': 'und so weiter',
        'etc.': 'et cetera',
        'ca.': 'circa',
        'bzw.': 'beziehungsweise',
        'ggf.': 'gegebenenfalls',
        'inkl.': 'inklusive',
        'exkl.': 'exklusive',
        'Tel.': 'Telefon',
        'Nr.': 'Nummer',
        'Str.': 'Straße',
        'zzgl.': 'zuzüglich',
    }
    for abbrev, expansion in abbreviations.items():
        text = text.replace(abbrev, expansion)

    # 9. Handle numbers with units (keep numbers, they're often fine)
    # Just ensure proper spacing
    text = re.sub(r'(\d+)\s*€', r'\1 Euro', text)
    text = re.sub(r'(\d+)\s*%', r'\1 Prozent', text)

    # 10. Remove control characters and unusual whitespace
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)

    # 11. Normalize whitespace (multiple spaces, tabs, newlines -> single space)
    text = re.sub(r'\s+', ' ', text).strip()

    # 12. Remove leading/trailing punctuation that shouldn't be there
    text = text.strip('.,;:!?-')
    text = text.strip()

    # Log if significant changes were made
    if text != original_text:
        logger.debug(f"TTS sanitization: '{original_text[:80]}...' -> '{text[:80]}...'")

    return text
