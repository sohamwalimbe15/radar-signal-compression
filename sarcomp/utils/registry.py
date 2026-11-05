"""Tiny name -> class/function registry used for every pluggable component."""


class Registry:
    def __init__(self, name):
        self.name = name
        self._items = {}

    def register(self, key):
        def deco(obj):
            if key in self._items:
                raise KeyError("%s: '%s' already registered" % (self.name, key))
            self._items[key] = obj
            return obj
        return deco

    def get(self, key):
        if key not in self._items:
            raise KeyError("%s: unknown '%s'. Available: %s" % (self.name, key, sorted(self._items)))
        return self._items[key]

    def build(self, key, *args, **kwargs):
        return self.get(key)(*args, **kwargs)

    def keys(self):
        return sorted(self._items)
