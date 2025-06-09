from configparser import ConfigParser

cfg = ConfigParser()
cfg.read("config.ini")

def save():
    with open("config.ini", "w") as f:
        cfg.write(f)
