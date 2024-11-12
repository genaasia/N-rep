import re


def extract_first_code_block(text: str) -> str:
    """extract code block contents"""
    pattern = r'```(?:sql|python|\w*)\n?(.*?)\n?```'
    matches = re.finditer(pattern, text, re.DOTALL)
    results = []
    for match in matches:
        content = match.group(1).strip()
        results.append(content)
    if len(results) == 0:
        return None
    return results[0]