class LazyFileReader(object):

    def __init__(self, path):
        self.path = path

    def __iter__(self):
        for line in open(self.path):
            yield line.strip().lower()