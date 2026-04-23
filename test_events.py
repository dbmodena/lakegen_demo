from llama_index.core.instrumentation.event_handlers import BaseEventHandler
from llama_index.core.instrumentation import get_dispatcher
import asyncio

class TestCapture(BaseEventHandler):
    @classmethod
    def class_name(cls) -> str:
        return "TestCapture"
    
    def handle(self, event) -> None:
        print(f"Event fired: {type(event).__name__}")
        if type(event).__name__ == "LLMChatEndEvent":
            print("  - LLMChatEndEvent response:", getattr(event, 'response', None))

dispatcher = get_dispatcher()
dispatcher.add_event_handler(TestCapture())

from llama_index.core.llms import ChatMessage
from llama_index.llms.ollama import Ollama

async def test():
    llm = Ollama(model="gemma4:26b", base_url="http://127.0.0.1:11434")
    try:
        res = await llm.achat([ChatMessage(role="user", content="Hi")])
    except Exception as e:
        print("LLM Error:", e)

asyncio.run(test())
