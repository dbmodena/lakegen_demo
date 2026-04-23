from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation import get_dispatcher
import asyncio
import re

class TestCapture(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "TestCapture"
    
    def handle(self, event) -> None:
        event_type = type(event).__name__
        if event_type == "LLMChatEndEvent":
            response = getattr(event, 'response', None)
            if response:
                msg = getattr(response, 'message', None)
                if msg:
                    content = getattr(msg, 'content', '')
                    blocks = getattr(msg, 'blocks', [])
                    print("--- LLMChatEndEvent ---")
                    print("Blocks:", [getattr(b, 'block_type', type(b)) for b in blocks])
                    print("Content length:", len(content) if content else 0)
                    if content and '<think>' in content:
                        print("Found <think> tags in content!")

dispatcher = get_dispatcher()
dispatcher.add_event_handler(TestCapture())

from llama_index.core.agent import ReActAgent
from llama_index.llms.ollama import Ollama

async def test():
    llm = Ollama(model="qwen2.5:14b", base_url="http://127.0.0.1:11434")  # Use a model we know might have think tags
    agent = ReActAgent.from_tools([], llm=llm, max_iterations=1)
    try:
        await agent.arun("Hi, can you think about the number 42?")
    except Exception as e:
        print("Agent Error:", e)

asyncio.run(test())
