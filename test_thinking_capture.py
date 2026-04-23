from llama_index.core.instrumentation.events.llm import LLMChatEndEvent
from llama_index.core.llms import ChatResponse, ChatMessage
from utils import ThinkingCapture

capture = ThinkingCapture()

# Mock a thinking block
class MockBlock:
    def __init__(self, block_type, content):
        self.block_type = block_type
        self.content = content

class MockMessage:
    def __init__(self):
        self.blocks = [MockBlock('thinking', 'This is a test thought')]

class MockResponse:
    def __init__(self):
        self.message = MockMessage()

# Mock the event
class MockEvent:
    def __init__(self):
        self.response = MockResponse()

event = MockEvent()
# Change the class name to simulate LLMChatEndEvent
event.__class__.__name__ = "LLMChatEndEvent"

capture.handle(event)
print("Captured parts:", capture.parts)
