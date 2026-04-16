"""Docchat: a document-aware AI agent chatbot powered by the Groq LLM API."""
import glob
import json
import os
import subprocess
import tempfile
from groq import Groq, BadRequestError
from dotenv import load_dotenv
import tools.calculate
import tools.ls
import tools.cat
import tools.grep
import tools.load_image

load_dotenv()

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The math expression to evaluate.",
                    }
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ls",
            "description": "List files in a directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default '.').",
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cat",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The file path to read.",
                    }
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for lines matching a regex pattern in files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File path or glob pattern to search.",
                    },
                },
                "required": ["pattern", "path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compact",
            "description": "Summarize the chat history to reduce token count.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "load_image",
            "description": "Load a local image file so the LLM can see it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the image file (JPEG, PNG, GIF, or WebP).",
                    }
                },
                "required": ["path"],
            },
        },
    },
]


def _speak(client, text):
    """
    Use Groq TTS to convert text to speech and play it.

    Saves a temporary WAV file then plays it with macOS `afplay` (or
    `aplay` on Linux).  Silently skips playback if neither is available.
    """
    try:
        response = client.audio.speech.create(
            model='canopylabs/orpheus-v1-english',
            voice='hannah',
            input=text,
            response_format='wav',
        )
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(response.read())
        for player in ('afplay', 'aplay'):
            try:
                subprocess.run([player, tmp_path], check=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                break
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue
    except Exception as e:
        print(f'[tts error] {e}')
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _execute_tool(name, args, messages=None):
    """
    Execute a named tool with the given arguments dict and return the result.

    >>> _execute_tool('calculate', {'expression': '2 + 2'})
    '4'
    >>> _execute_tool('ls', {'path': 'test_data'})
    'binary.bin\\nhello.txt\\nnumbers.txt\\ntest.png\\nutf16.txt'
    >>> _execute_tool('cat', {'path': 'test_data/hello.txt'})
    'Hello, World!'
    >>> _execute_tool('grep', {'pattern': 'Hello', 'path': 'test_data/hello.txt'})
    'Hello, World!'
    >>> _execute_tool('ls', {})  # doctest: +ELLIPSIS
    '...'
    >>> _execute_tool('load_image', {'path': 'x'})
    'Error: load_image requires access to the messages list'
    >>> _execute_tool('load_image', {'path': 'test_data/hello.txt'}, messages=[])
    'Error: unsupported image type: text/plain'
    >>> _execute_tool('unknown', {})
    'Error: unknown tool unknown'
    """
    if name == 'calculate':
        return tools.calculate.calculate(args['expression'])
    elif name == 'ls':
        return tools.ls.ls(args.get('path', '.'))
    elif name == 'cat':
        return tools.cat.cat(args['path'])
    elif name == 'grep':
        return tools.grep.grep(args['pattern'], args['path'])
    elif name == 'load_image':
        if messages is None:
            return 'Error: load_image requires access to the messages list'
        return tools.load_image.load_image(args['path'], messages)
    return f'Error: unknown tool {name}'


class Chat:
    '''
    An AI chat agent powered by the Groq LLM API that maintains conversation
    history and can invoke tools to explore files and perform calculations.
    Responds in a pirate-themed style.

    >>> chat = Chat()
    >>> _ = chat.send_message('my name is bob', temperature=0.0)
    >>> _ = chat.send_message('what is my name?', temperature=0.0)

    >>> chat2 = Chat()
    >>> _ = chat2.send_message('what is my name?', temperature=0.0)

    >>> len(chat.messages)
    5
    >>> summary = chat.compact()
    >>> isinstance(summary, str)
    True
    >>> len(chat.messages)
    2

    >>> Chat(debug=True).debug
    True
    >>> Chat(use_tools=False)._tools is None
    True
    >>> Chat(tts=True).tts
    True

    >>> class FakeMsg:
    ...     role = 'user'
    ...     content = 'hello'
    >>> chat3 = Chat()
    >>> chat3.messages.append(FakeMsg())
    >>> isinstance(chat3.compact(), str)
    True
    '''

    def __init__(self, debug=False, use_tools=True, tts=False):
        """Initialize the chat agent with optional debug, tool, and TTS flags."""
        self.debug = debug
        self.tts = tts
        self._tools = TOOLS if use_tools else None
        self.client = Groq()
        self.messages = [
            {
                "role": "system",
                "content": (
                    "Write the output in 1-2 sentences. Talk like pirate. "
                    "Only use tools when the user explicitly asks you to list "
                    "files, read files, search files, or calculate something. "
                    "Never use tools for normal conversation."
                ),
            }
        ]

    def compact(self):
        """Summarize the chat history and replace messages with the summary."""
        subagent = Chat(use_tools=False)
        history_parts = []
        for m in self.messages:
            if isinstance(m, dict):
                role = m.get('role', '')
                content = m.get('content', '')
            else:
                role = getattr(m, 'role', '')
                content = getattr(m, 'content', '') or ''
            if role != 'system' and content:
                history_parts.append(f'{role}: {content}')
        history = '\n'.join(history_parts)
        summary = subagent.send_message(
            f'Summarize this chat history in 1-5 lines:\n{history}',
            temperature=0.0
        )
        self.messages = [
            self.messages[0],
            {'role': 'assistant', 'content': f'[Summary]: {summary}'}
        ]
        return summary

    def send_message(self, message, temperature=0.8):
        """
        Send a message and return the response, executing any tool calls first.
        """
        self.messages.append({'role': 'user', 'content': message})
        while True:
            # Use vision-capable model when any message contains image content
            has_image = any(
                isinstance(m.get('content'), list)
                for m in self.messages
                if isinstance(m, dict)
            )
            model = 'meta-llama/llama-4-scout-17b-16e-instruct' if has_image else 'llama-3.1-8b-instant'
            kwargs = {
                'messages': self.messages,
                'model': model,
                'temperature': temperature,
            }
            if self._tools:
                kwargs['tools'] = self._tools
            try:
                completion = self.client.chat.completions.create(**kwargs)
            except BadRequestError as e:
                if 'tool_use_failed' in str(e):
                    completion = self.client.chat.completions.create(**kwargs)
                else:
                    raise
            choice = completion.choices[0]
            if choice.finish_reason == 'tool_calls':
                self.messages.append(choice.message)
                for tool_call in choice.message.tool_calls:
                    name = tool_call.function.name
                    call_args = json.loads(tool_call.function.arguments)
                    if self.debug:
                        arg_str = ' '.join(str(v) for v in call_args.values())
                        print(f'[tool] /{name} {arg_str}'.rstrip())
                    if name == 'compact':
                        result = self.compact()
                    else:
                        result = _execute_tool(name, call_args, messages=self.messages)
                    self.messages.append({
                        'role': 'tool',
                        'tool_call_id': tool_call.id,
                        'content': result,
                    })
            else:
                result = choice.message.content
                self.messages.append(
                    {'role': 'assistant', 'content': result}
                )
                if self.tts:
                    _speak(self.client, result)
                return result


def _handle_slash_command(user_input, chat=None):
    """
    Parse and execute a slash command, returning the result as a string.

    >>> _handle_slash_command('/ls test_data')
    'binary.bin\\nhello.txt\\nnumbers.txt\\ntest.png\\nutf16.txt'
    >>> _handle_slash_command('/cat test_data/hello.txt')
    'Hello, World!'
    >>> _handle_slash_command('/calculate 6 * 7')
    '42'
    >>> _handle_slash_command('/grep Hello test_data/hello.txt')
    'Hello, World!'
    >>> _handle_slash_command('/')
    'Error: empty command'
    >>> _handle_slash_command('/cat')
    'Error: cat requires a file path'
    >>> _handle_slash_command('/grep Hello')
    'Error: grep requires a pattern and file path'
    >>> _handle_slash_command('/compact')
    'Error: compact requires an active chat session'
    >>> isinstance(_handle_slash_command('/compact', chat=Chat()), str)
    True
    >>> _handle_slash_command('/load_image')
    'Error: load_image requires a file path'
    >>> _handle_slash_command('/load_image nonexistent.png')
    'Error: file not found: nonexistent.png'
    >>> _handle_slash_command('/load_image test_data/hello.txt')
    'Error: load_image requires an active chat session'
    >>> _handle_slash_command('/load_image test_data/test.png', chat=Chat())
    'Image loaded: test_data/test.png'
    >>> _handle_slash_command('/unknown arg')
    'Error: unknown command unknown'
    """
    parts = user_input[1:].split()
    if not parts:
        return 'Error: empty command'
    cmd = parts[0]
    args = parts[1:]
    if cmd == 'calculate':
        return tools.calculate.calculate(' '.join(args))
    elif cmd == 'ls':
        return tools.ls.ls(args[0] if args else '.')
    elif cmd == 'cat':
        if not args:
            return 'Error: cat requires a file path'
        return tools.cat.cat(args[0])
    elif cmd == 'grep':
        if len(args) < 2:
            return 'Error: grep requires a pattern and file path'
        return tools.grep.grep(args[0], args[1])
    elif cmd == 'compact':
        if chat is None:
            return 'Error: compact requires an active chat session'
        return chat.compact()
    elif cmd == 'load_image':
        if not args:
            return 'Error: load_image requires a file path'
        if not os.path.isfile(args[0]):
            return f'Error: file not found: {args[0]}'
        if chat is None:
            return 'Error: load_image requires an active chat session'
        return tools.load_image.load_image(args[0], chat.messages)
    return f'Error: unknown command {cmd}'


def _make_completer():
    """
    Create a readline tab completer for slash commands and file paths.

    >>> completer = _make_completer()
    >>> completer('/l', 0)
    '/load_image'
    >>> completer('/l', 1)
    '/ls'
    >>> completer('/l', 2) is None
    True
    >>> completer('/ca', 0)
    '/calculate'
    >>> completer('/ca', 1)
    '/cat'
    >>> completer('/ca', 2) is None
    True
    >>> completer('test_data/h', 0)
    'test_data/hello.txt'
    >>> completer('nonexistent_path_xyz', 0) is None
    True
    """
    commands = ['calculate', 'cat', 'compact', 'grep', 'load_image', 'ls']

    def completer(text, state):
        if text.startswith('/'):
            prefix = text[1:]
            matches = ['/' + c for c in commands if c.startswith(prefix)]
        else:
            paths = sorted(glob.glob(text + '*'))
            matches = [p + ('/' if os.path.isdir(p) else '') for p in paths]
        try:
            return matches[state]
        except IndexError:
            return None

    return completer


def repl(temperature=0.8, debug=False, tts=False):
    """
    Run the interactive REPL supporting slash commands and LLM chat.

    #monkey patch doctest
    >>> def monkey_input(prompt, user_inputs=['/cat test_data/hello.txt', '/calculate 6 * 7']):
    ...     try:
    ...         user_input = user_inputs.pop(0)
    ...         print(f'{prompt}{user_input}')
    ...         return user_input
    ...     except IndexError:
    ...         raise KeyboardInterrupt
    >>> import builtins
    >>> builtins.input = monkey_input
    >>> repl(temperature=0.0)
    chat> /cat test_data/hello.txt
    Hello, World!
    chat> /calculate 6 * 7
    42
    <BLANKLINE>

    #monkey patch doctest for debug mode (slash commands unaffected by debug)
    >>> def monkey_input_debug(prompt, user_inputs=['/cat test_data/hello.txt']):
    ...     try:
    ...         user_input = user_inputs.pop(0)
    ...         print(f'{prompt}{user_input}')
    ...         return user_input
    ...     except IndexError:
    ...         raise KeyboardInterrupt
    >>> builtins.input = monkey_input_debug
    >>> repl(temperature=0.0, debug=True)
    chat> /cat test_data/hello.txt
    Hello, World!
    <BLANKLINE>

    #monkey patch doctest for LLM chat path
    >>> def monkey_input_chat(prompt, user_inputs=['say exactly the word: Arrr']):
    ...     try:
    ...         user_input = user_inputs.pop(0)
    ...         print(f'{prompt}{user_input}')
    ...         return user_input
    ...     except IndexError:
    ...         raise KeyboardInterrupt
    >>> builtins.input = monkey_input_chat
    >>> repl(temperature=0.0)  # doctest: +ELLIPSIS
    chat> say exactly the word: Arrr
    ...
    <BLANKLINE>
    """
    try:
        import readline
        readline.set_completer_delims(' \t\n')
        readline.set_completer(_make_completer())
        readline.parse_and_bind('tab: complete')
    except ImportError:
        pass
    chat = Chat(debug=debug, tts=tts)
    try:
        while True:
            user_input = input('chat> ')
            if user_input.startswith('/'):
                print(_handle_slash_command(user_input, chat=chat))
            else:
                response = chat.send_message(user_input, temperature=temperature)
                print(response)
    except (KeyboardInterrupt, EOFError):
        print()


def main():
    """
    Entry point for the chat CLI, supporting an optional message and --debug flag.
    """
    import argparse
    parser = argparse.ArgumentParser(description='Docchat: chat with documents')
    parser.add_argument('message', nargs='?', help='Message to send to the LLM')
    parser.add_argument('--debug', action='store_true', help='Print tool calls')
    parser.add_argument('--tts', action='store_true', help='Read responses aloud using TTS')
    args = parser.parse_args()
    if args.message:
        chat = Chat(debug=args.debug, tts=args.tts)
        print(chat.send_message(args.message))
    else:
        repl(debug=args.debug, tts=args.tts)


if __name__ == '__main__':
    main()
