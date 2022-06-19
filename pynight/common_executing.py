import executing
import inspect
from textwrap import dedent

class Source(executing.Source):
    def get_text_with_indentation(self, node):
        result = self.asttokens().get_text(node)
        if '\n' in result:
            result = ' ' * node.first_token.start[1] + result
            result = dedent(result)
        result = result.strip()
        return result

def get_with_source(val):
    callFrame = inspect.currentframe().f_back
    callNode = Source.executing(callFrame).node
    source = Source.for_frame(callFrame)
    return val, source.get_text_with_indentation(callNode.args[0])
