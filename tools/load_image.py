"""Tool for loading a local image into the chat message history."""
import base64
import mimetypes
import os


def load_image(path, messages):
    """
    Load a local image file and inject it into the messages list as a
    vision-compatible user message.  Returns a confirmation string.

    Because tool results must be plain text, this function directly appends
    to the shared messages list rather than returning the image data.

    >>> import tempfile, os
    >>> tmp = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
    >>> _ = tmp.write(b'\\x89PNG\\r\\n\\x1a\\n' + b'\\x00' * 8)
    >>> tmp.close()
    >>> msgs = []
    >>> result = load_image(tmp.name, msgs)
    >>> result.startswith('Image loaded:')
    True
    >>> len(msgs) == 1
    True
    >>> msgs[0]['role']
    'user'
    >>> os.unlink(tmp.name)
    >>> load_image('nonexistent_xyz.png', [])
    'Error: file not found: nonexistent_xyz.png'
    >>> tmp2 = tempfile.NamedTemporaryFile(suffix='.txt', delete=False)
    >>> tmp2.close()
    >>> load_image(tmp2.name, [])
    'Error: unsupported image type: text/plain'
    >>> os.unlink(tmp2.name)
    """
    if not os.path.isfile(path):
        return f'Error: file not found: {path}'
    mime_type, _ = mimetypes.guess_type(path)
    if mime_type not in ('image/jpeg', 'image/png', 'image/gif', 'image/webp'):
        return f'Error: unsupported image type: {mime_type}'
    with open(path, 'rb') as f:
        data = base64.standard_b64encode(f.read()).decode('utf-8')
    messages.append({
        'role': 'user',
        'content': [
            {
                'type': 'image_url',
                'image_url': {
                    'url': f'data:{mime_type};base64,{data}',
                },
            }
        ],
    })
    return f'Image loaded: {path}'
