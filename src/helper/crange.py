class crange:
    def __init__(self, start, stop, step=None, modulo=None):
        if step == 0:
            raise ValueError('crange() arg 3 must not be zero')

        if step is None and modulo is None:
            self.start = 0
            self.stop = start
            self.step = 1
            self.modulo = stop
        else:
            self.start = start
            self.stop = stop
            if modulo is None:
                self.step = 1
                self.modulo = step
            else:
                self.step = step
                self.modulo = modulo

    def __iter__(self):
        n = self.start
        if n > self.stop:
            while n < self.modulo:
                yield n
                n += 1
            n = 0
        while n < self.stop:
            yield n
            n += 1

    def __contains__(self, n):
        if self.start >= self.stop:
            return self.start <= n < self.modulo or 0 <= n < self.stop
        else:
            return self.start <= n < self.stop


if __name__=='__main__':
    print(list(crange(start=40, stop=20, modulo=100)))
