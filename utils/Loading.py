import itertools
import sys
import threading
import time


class Loading:
    def __init__(self, loading_message="Loading"):
        self.loading_message = loading_message
        self.done_loading = False
        self._loading()

    def _loading(self):
        def animate():
            for c in itertools.cycle(["|", "/", "-", "\\"]):
                if self.done_loading:
                    break
                sys.stdout.write(f"\r{self.loading_message} " + c + " ")
                sys.stdout.write('\033[2K\033[1G')
                time.sleep(0.1)

        t = threading.Thread(target=animate)
        t.daemon = True
        t.start()

    def stop_loading(self):
        self.done_loading = True
