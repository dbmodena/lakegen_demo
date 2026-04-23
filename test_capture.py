import asyncio
from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.core.llms import ChatResponse, ChatMessage
from utils import ThinkingCapture

capture = ThinkingCapture()

# Create a mock event that mimics LlamaIndex structure
class MockBlock:
    def __init__(self, block_type, content):
        self.block_type = block_type
        self.content = content

class MockMessage:
    def __init__(self):
        # Extremely long string to prove it isn't truncated
        self.blocks = [MockBlock('thinking', 'This is a very long thinking block ' * 100)]
        self.content = '<think>' + 'Raw thinking block ' * 100 + '</think>'

class MockResponse:
    def __init__(self):
        self.message = MockMessage()

class MockEvent:
    def __init__(self):
        self.response = MockResponse()

event = MockEvent()
event.__class__.__name__ = "LLMChatEndEvent"

capture.handle(event)

print(f"Captured parts count: {len(capture.parts)}")
if capture.parts:
    print(f"Length of first captured part: {len(capture.parts[0])}")
    print(f"Length of second captured part: {len(capture.parts[1])}")
