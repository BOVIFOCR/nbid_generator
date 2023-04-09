import os
import sys
import logging


stdout_handler = logging.FileHandler('stdout.log', 'w')
stdout_handler.setLevel(logging.DEBUG if 'DEBUGGING' in os.environ else logging.INFO)

# handler = logging.StreamHandler(sys.stderr)
stderr_handler = logging.FileHandler('stderr.log', 'w')
stderr_handler.setLevel(logging.ERROR)

logging.basicConfig(
    handlers=[stdout_handler, stderr_handler]
)

logging.getLogger('matplotlib').setLevel(logging.CRITICAL)
