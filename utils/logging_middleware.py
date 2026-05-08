import time, sys
import logging

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware


log = logging.getLogger(__name__)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Log request details
        log.info(f"Received request: {request.method} {request.url}")
        
        # Process the request
        response = await call_next(request)
        
        process_time = time.time() - start_time
        formatted_process_time = '{0:.2f}'.format(process_time * 1000)
        
        # Log response details
        log.info(
            f"Completed response: status_code={response.status_code} "
            f"method={request.method} url={request.url} "
            f"process_time={formatted_process_time}ms"
        )
        
        return response
