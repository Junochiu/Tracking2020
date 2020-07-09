"""
Example Python implementation of the a static tracker
"""

from trax import *
import time

region = None
delay = 0
with Server([Region.RECTANGLE], [Image.PATH], verbose=True) as server:
    while True:
        request = server.wait()
        if request.type in [TraxStatus.QUIT, TraxStatus.ERROR]:
            break
        if request.type == TraxStatus.INITIALIZE:
            region = request.region
            delay = float(request.properties.get("time_wait", "0"))
        server.status(region)
        if delay > 0:
            time.sleep(delay)
