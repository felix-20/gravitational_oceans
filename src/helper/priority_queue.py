class GOPriorityQueue:

    def __init__(self) -> None:
        self.queue = []

    def insert(self, element: tuple[float, float]) -> tuple[int, bool]:
        if not self.queue:
            self.queue += [element]
            return (0, False)

        for i in range(len(self.queue)):
            item = self.queue[i]
            # check for interval intersection
            if element[0] < item[1] and item[0] < element[1]:
                return (i, True)

            if element[1] < item[0]:
                # the new element needs to be inserted before the current item
                self.queue = self.queue[:i] + [element] + self.queue[i:]
                return (i, False)
        # the element is larger than all already inserted elements
        self.queue += [element]
        return (len(self.queue) - 1, False)

    def __str__(self) -> str:
        return str(self.queue)

if __name__ == '__main__':
    a = GOPriorityQueue()
    print(a.insert((1.01, 2.02)))
    print(a.insert((4.02, 5.02)))
    print(a.insert((0.3, 0.8)))
    print(a.insert((0.0, 0.02)))
    print(a.insert((2.5, 3.0)))
    print(a)
