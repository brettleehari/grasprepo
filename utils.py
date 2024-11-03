import itertools
import sys

def dump(*args):
    print("Dumping values:", args)
    print(*args)

def filter_important_files(filenames):
    # Implement the logic to filter important files
    print(f"Filtering important files from: {filenames}")
    return filenames  # Modify as needed

class Spinner:
    spinner_chars = itertools.cycle(["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"])

    def __init__(self, message=""):
        self.message = message
        self.running = False

    def step(self):
        print("Spinner step")
        if not self.running:
            self.running = True
            sys.stdout.write(self.message + " ")
        sys.stdout.write(next(self.spinner_chars))
        sys.stdout.flush()
        sys.stdout.write("\b")

    def end(self):
        print("Spinner end")
        if self.running:
            sys.stdout.write(" Done\n")
            sys.stdout.flush()
            self.running = False
