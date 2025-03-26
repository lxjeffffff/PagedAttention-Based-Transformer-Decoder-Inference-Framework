# web/sse_utils.py
import json

def stream_sse(generator_func):
    from flask import Response
    def event_stream():
        for data in generator_func():
            yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        yield 'event: done\ndata: {"status": "end"}\n\n'
    return Response(event_stream(), mimetype='text/event-stream')
