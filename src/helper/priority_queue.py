class GOPriorityQueue:

    def __init__(self) -> None:
        self.queue = []

    def insert(self, element: tuple[float, float]) -> int:
        if not self.queue:
            self.queue += [element]
            return 0
        has_been_inserted = False
        for i in range(len(self.queue)):
            item = self.queue[i]

            # check for interval intersection
            if element[0] < item[1] and item[0] < element[1]:
                raise ValueError(f'intervall intersection: {element}, {item}')

            if element[1] < item[0]:
                # the new element needs to be inserted before the current item
                self.queue = self.queue[:i] + [element] + self.queue[i+1:]
                return i

        if not has_been_inserted:
            # the element is larger than all already inserted elements
            self.queue += [element]
            return len(self.queue) - 1

    def __str__(self) -> str:
        return str(self.queue)

if __name__ == '__main__':
    a = GOPriorityQueue()
    a.insert((1.01, 2.02))
    a.insert((4.02, 5.02))
    a.insert((0.3, 0.8))
    a.insert((0.0, 0.02))
