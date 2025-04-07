import threading


class TokenCounter:
    def __init__(self):
        """A thread-safe counter for tracking token usage in LLM API calls.
        
        All operations are protected by a threading lock to ensure thread safety
        when updating counts from multiple threads.
        """
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0
        self.lock = threading.Lock()
    
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
