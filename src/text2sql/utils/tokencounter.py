import threading


class TokenCounter:
    def __init__(self, initial_counts=None):
        """A thread-safe counter for tracking token usage in LLM API calls.

        All operations are protected by a threading lock to ensure thread safety
        when updating counts from multiple threads.

        Args:
            initial_counts (dict, optional): Dictionary with initial values for token counts.
                Should have keys: 'prompt_tokens', 'completion_tokens', 'total_tokens', 'call_count'.
                If not provided, all counts start at 0.
        """
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0
        self.lock = threading.Lock()
        
        if initial_counts is not None:
            self.update_counts(initial_counts)

    def update_counts(self, counts: dict) -> None:
        """Update the counter with values from the provided dictionary.

        Args:
            counts (dict): Dictionary with values to add to the current counts.
                Should have keys: 'prompt_tokens', 'completion_tokens', 'total_tokens', 'call_count'.
        """
        with self.lock:
            self.prompt_tokens += counts.get('prompt_tokens', 0)
            self.completion_tokens += counts.get('completion_tokens', 0)
            self.total_tokens += counts.get('total_tokens', 0)
            self.call_count += counts.get('call_count', 0)

    def add_token_counts(self, prompt_tokens, completion_tokens) -> None:
        """Add token counts to the counter and increase call count.

        Args:
            prompt_tokens (int): The number of prompt tokens.
            completion_tokens (int): The number of completion tokens.
        """
        with self.lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.total_tokens += prompt_tokens + completion_tokens
            self.call_count += 1

    def get_counts(self) -> dict:
        """Get the current token counts.

        Returns:
            A dictionary containing all token counts and call count.
        """
        with self.lock:
            return {
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "total_tokens": self.total_tokens,
                "call_count": self.call_count,
            }


class CharacterCounter:
    def __init__(self, initial_counts=None):
        """A thread-safe counter for tracking character usage in LLM API calls.

        All operations are protected by a threading lock to ensure thread safety
        when updating counts from multiple threads.

        Args:
            initial_counts (dict, optional): Dictionary with initial values for character counts.
                Should have keys: 'characters', 'call_count'.
                If not provided, all counts start at 0.
        """
        self.characters: int = 0
        self.call_count: int = 0
        self.lock = threading.Lock()
        
        if initial_counts is not None:
            self.update_counts(initial_counts)

    def update_counts(self, counts: dict) -> None:
        """Update the counter with values from the provided dictionary.

        Args:
            counts (dict): Dictionary with values to add to the current counts.
                Should have keys: 'characters', 'call_count'.
        """
        with self.lock:
            self.characters += counts.get('characters', 0)
            self.call_count += counts.get('call_count', 0)

    def add_character_counts(self, characters: int) -> None:
        """Add token counts to the counter and increase call count.

        Args:
            characters (int): The number of (input) characters.
        """
        with self.lock:
            self.characters += characters
            self.call_count += 1

    def get_counts(self) -> dict:
        """Get the current character counts.

        Returns:
            A dictionary containing character count and call count.
        """
        with self.lock:
            return {
                "characters": self.characters,
                "call_count": self.call_count,
            }
