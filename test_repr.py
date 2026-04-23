from llama_index.core.llms import ChatMessage
import sys
# It's an internal class, but let's check
try:
    from llama_index.core.llms.callbacks import ThinkingBlock
    print(repr(ThinkingBlock(block_type='thinking', content='A' * 200)))
except ImportError:
    pass

try:
    from llama_index.core.llms import TextBlock
    # Let's create a ChatMessage
    msg = ChatMessage(role="assistant", content="A" * 200)
    print(repr(msg))
except Exception as e:
    print(e)
