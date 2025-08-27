# filepath: d:\Sfe talk last git\tests\test_server.py
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from server import check_profanity, check_cyberbullying

def test_check_profanity_clean():
    result, prob = check_profanity("Hello, how are you?")
    assert result is False
    assert prob == 0.0

def test_check_profanity_bad():
    result, prob = check_profanity("shit")
    assert result is True
    assert prob == 1.0

def test_check_cyberbullying_clean():
    is_safe, reason = check_cyberbullying("Have a nice day!")
    assert is_safe is True
    assert "approved" in reason.lower()

def test_check_cyberbullying_profanity():
    is_safe, reason = check_cyberbullying("fuck you")
    assert is_safe is False
    assert "profanity" in reason.lower()