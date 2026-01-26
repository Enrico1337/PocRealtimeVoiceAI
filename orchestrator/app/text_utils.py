"""
Text utilities for TTS preprocessing.

This module provides text sanitization to prepare LLM output for TTS synthesis,
preventing gibberish/fantasy language output from both Chatterbox and Coqui engines.
"""

import re
import unicodedata
import logging

logger = logging.getLogger(__name__)


def normalize_german_phone_numbers(text: str) -> str:
    """Convert phone numbers to digit-by-digit German speech.

    Examples:
        "0800-1234567" → "null acht null null eins zwei drei vier fünf sechs sieben"
        "+49 123 456789" → "plus vier neun eins zwei drei vier fünf sechs sieben acht neun"
    """
    digit_words = {
        '0': 'null', '1': 'eins', '2': 'zwei', '3': 'drei', '4': 'vier',
        '5': 'fünf', '6': 'sechs', '7': 'sieben', '8': 'acht', '9': 'neun'
    }

    def replace_phone(match: re.Match) -> str:
        phone = match.group(0)
        # Keep "plus" for international prefix
        result = phone.replace('+', 'plus ')
        # Replace each digit with German word
        for digit, word in digit_words.items():
            result = result.replace(digit, word + ' ')
        # Remove separators and clean up
        result = re.sub(r'[-/\s]+', ' ', result)
        return result.strip()

    # Match phone number patterns:
    # - Starting with + or 0
    # - Containing digits, spaces, hyphens, slashes
    # - At least 6 digits total
    phone_pattern = r'(?<!\d)[\+]?[0-9][\d\s\-/]{5,}[\d](?!\d)'

    return re.sub(phone_pattern, replace_phone, text)


def normalize_german_times(text: str) -> str:
    """Convert time formats to natural German speech.

    Examples:
        "9:00" → "neun Uhr"
        "9:00 Uhr" → "neun Uhr"
        "10:30" → "zehn Uhr dreißig"
        "14:45 Uhr" → "vierzehn Uhr fünfundvierzig"
    """
    hour_words = {
        0: 'null', 1: 'ein', 2: 'zwei', 3: 'drei', 4: 'vier', 5: 'fünf',
        6: 'sechs', 7: 'sieben', 8: 'acht', 9: 'neun', 10: 'zehn',
        11: 'elf', 12: 'zwölf', 13: 'dreizehn', 14: 'vierzehn', 15: 'fünfzehn',
        16: 'sechzehn', 17: 'siebzehn', 18: 'achtzehn', 19: 'neunzehn',
        20: 'zwanzig', 21: 'einundzwanzig', 22: 'zweiundzwanzig', 23: 'dreiundzwanzig'
    }

    minute_words = {
        0: '', 5: 'fünf', 10: 'zehn', 15: 'fünfzehn', 20: 'zwanzig',
        25: 'fünfundzwanzig', 30: 'dreißig', 35: 'fünfunddreißig',
        40: 'vierzig', 45: 'fünfundvierzig', 50: 'fünfzig', 55: 'fünfundfünfzig'
    }

    def replace_time(match: re.Match) -> str:
        hour = int(match.group(1))
        minute = int(match.group(2))

        if hour > 23:
            return match.group(0)  # Invalid hour, keep original

        hour_word = hour_words.get(hour, str(hour))

        if minute == 0:
            return f"{hour_word} Uhr"
        elif minute in minute_words:
            return f"{hour_word} Uhr {minute_words[minute]}"
        else:
            # For non-standard minutes, use digit-based approach
            return f"{hour_word} Uhr {minute}"

    # Match HH:MM or H:MM, optionally followed by " Uhr"
    time_pattern = r'\b(\d{1,2}):(\d{2})(\s*Uhr)?(?!\d)'

    return re.sub(time_pattern, replace_time, text)


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

    # 8b. Normalize German phone numbers (read digit by digit)
    text = normalize_german_phone_numbers(text)

    # 8c. Normalize German time formats
    text = normalize_german_times(text)

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
