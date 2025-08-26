
import os
import time
import logging
from typing import List, Optional
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential

class APIKeyManager:
    """Manages multiple API keys with rotation and quota management."""
    
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_key_index = 0
        self.key_usage_count = {key: 0 for key in api_keys}
        self.key_last_used = {key: 0 for key in api_keys}
        self.key_blocked_until = {key: 0 for key in api_keys}
        self.requests_per_minute_per_key = 60  # Conservative limit
        self.logger = logging.getLogger(__name__)
        
        # Initialize first key
        self._configure_current_key()
    
    def _configure_current_key(self):
        """Configure the current API key."""
        current_key = self.api_keys[self.current_key_index]
        genai.configure(api_key=current_key)
        self.logger.info(f"Configured API key {self.current_key_index + 1}/{len(self.api_keys)}")
    
    def get_current_key(self) -> str:
        """Get the current active API key."""
        return self.api_keys[self.current_key_index]
    
    def rotate_key(self, reason: str = "rotation"):
        """Rotate to the next available API key."""
        old_index = self.current_key_index
        
        # Try to find a non-blocked key
        for i in range(len(self.api_keys)):
            next_index = (self.current_key_index + 1 + i) % len(self.api_keys)
            next_key = self.api_keys[next_index]
            
            # Check if key is not blocked
            if time.time() > self.key_blocked_until.get(next_key, 0):
                self.current_key_index = next_index
                self._configure_current_key()
                self.logger.warning(f"Rotated from key {old_index + 1} to key {self.current_key_index + 1} ({reason})")
                return True
        
        # All keys are blocked - wait for the next available one
        min_wait_time = min(self.key_blocked_until.values()) - time.time()
        if min_wait_time > 0:
            self.logger.warning(f"All keys quota-limited. Waiting {min_wait_time:.1f}s for next available key")
            time.sleep(min_wait_time + 1)
            return self.rotate_key("quota_recovery")
        
        return False
    
    def handle_quota_error(self, error_msg: str):
        """Handle quota/rate limit errors by blocking current key and rotating."""
        current_key = self.get_current_key()
        
        # Block current key for 60 seconds
        self.key_blocked_until[current_key] = time.time() + 60
        
        # Determine block duration based on error type
        if "quota" in error_msg.lower():
            block_duration = 300  # 5 minutes for quota errors
        elif "rate" in error_msg.lower():
            block_duration = 60   # 1 minute for rate limit errors
        else:
            block_duration = 120  # 2 minutes for other errors
        
        self.key_blocked_until[current_key] = time.time() + block_duration
        
        self.logger.warning(f"Key {self.current_key_index + 1} blocked for {block_duration}s due to: {error_msg}")
        
        # Rotate to next available key
        self.rotate_key("quota_error")
    
    def rate_limit(self):
        """Implement per-key rate limiting."""
        current_key = self.get_current_key()
        current_time = time.time()
        
        # Check if we need to wait
        last_used = self.key_last_used.get(current_key, 0)
        min_interval = 60.0 / self.requests_per_minute_per_key
        
        time_since_last = current_time - last_used
        if time_since_last < min_interval:
            sleep_time = min_interval - time_since_last
            time.sleep(sleep_time)
        
        # Update usage tracking
        self.key_last_used[current_key] = time.time()
        self.key_usage_count[current_key] += 1
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def make_api_call(self, model_name: str, prompt: str, **kwargs):
        """Make API call with automatic key rotation on errors."""
        self.rate_limit()
        
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt, **kwargs)
            
            if not response.text:
                raise ValueError("Empty response from model")
            
            return {
                "text": response.text,
                "model": model_name,
                "api_key_index": self.current_key_index + 1
            }
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle quota/rate limit errors
            if any(keyword in error_msg for keyword in ["quota", "rate", "limit", "429"]):
                self.handle_quota_error(str(e))
                # Retry with new key
                raise
            else:
                # Other errors - log and re-raise
                self.logger.error(f"API call failed: {e}")
                raise
    
    def get_usage_stats(self) -> dict:
        """Get usage statistics for all keys."""
        return {
            "current_key": self.current_key_index + 1,
            "usage_per_key": {
                f"key_{i+1}": count 
                for i, count in enumerate(self.key_usage_count.values())
            },
            "blocked_keys": [
                f"key_{i+1}" 
                for i, (key, blocked_until) in enumerate(self.key_blocked_until.items())
                if time.time() < blocked_until
            ]
        }
