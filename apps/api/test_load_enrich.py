import watchdog.events
import watchdog.observers
import time

class FileEventHandler(watchdog.events.PatternMatchingEventHandler):
    def on_created(self, event):
        print(f"File created: {event.src_path}")

    def on_deleted(self, event):
        print(f"File deleted: {event.src_path}")

    def on_modified(self, event):
        print(f"File modified: {event.src_path}")

    def on_moved(self, event):
        print(f"File moved: {event.src_path}")

if __name__ == "__main__":
    observer = watchdog.observers.Observer()
    observer.schedule(FileEventHandler(), path=".local_manifest", recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()