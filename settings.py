import json
import os
import os.path

class Settings():
    def __init__(self):
        path = 'settings.json'
        settings = {}
        if os.path.exists(path):
            with open(path) as f:
                settings = json.load(f)

        for key in settings:
            self.__dict__[key] = settings[key]
