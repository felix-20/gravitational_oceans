from __future__ import annotations

class GOPriorityQueue:

    def __init__(self) -> None:
        self.sorted_list = []
    
    def insert(self, element: tuple[float, float]) -> int:
        for i in range(len(self.sorted_list)):
            item = self.sorted_list[i]

            if element[0] < item[1] and item[0] < element[1]:
                raise ValueError('intervall intersection')
            
            if element[1] < item[0]:
                self.sorted_list = self.sorted_list[:i] + [element] + self.sorted_list[i+1:]

