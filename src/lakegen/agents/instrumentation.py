import re
from typing import List
from pydantic import Field
from llama_index.core.instrumentation.event_handlers import BaseEventHandler

class ThinkingCapture(BaseEventHandler):
    parts: List[str] = Field(default_factory=list)

    @classmethod
    def class_name(cls) -> str:
        return "ThinkingCapture"

    def handle(self, event) -> None:
        event_type = type(event).__name__
        
        def extract_thinking(msg):
            # Extract from LlamaIndex structured thinking blocks
            for block in getattr(msg, 'blocks', []):
                if getattr(block, 'block_type', '') == 'thinking':
                    content = getattr(block, 'content', '')
                    if content and content not in self.parts:
                        self.parts.append(content)
                        
            # Extract from raw text using <think> tags (fallback for some Ollama models)
            content = getattr(msg, 'content', '')
            if content and isinstance(content, str):
                matches = re.findall(r'<think>(.*?)</think>', content, re.DOTALL | re.IGNORECASE)
                for match in matches:
                    text = match.strip()
                    if text and text not in self.parts:
                        self.parts.append(text)
                        
        # Capture from LLMChatEndEvent
        if event_type == "LLMChatEndEvent":
            response = getattr(event, 'response', None)
            if response:
                msg = getattr(response, 'message', None)
                if msg:
                    extract_thinking(msg)
                                
        # Capture from AgentRunStepEndEvent or Workflow Step Events
        elif event_type in ["AgentRunStepEndEvent", "StepEndEvent", "AgentOutput"]:
            output = getattr(event, 'step_output', None) or getattr(event, 'output', None) or event
            if output is None:
                return
            msg = getattr(output, 'output', None) or getattr(output, 'response', None)
            if msg:
                extract_thinking(msg)
