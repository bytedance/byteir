import os
import json
import fcntl
import tempfile

from shutil import copyfile, copymode

HOME_DIR = os.getenv("HOME")
if HOME_DIR is None:
    HOME_DIR = tempfile.gettempdir()
DEFAULT_CACHE_DIR = os.path.join(HOME_DIR, ".byteir", "ait_cache")
CACHE_FILE_NAME = "ait_global_cache.json"
IDX_KEY = "byteir_ait_cache_auto_increment_idx"

class AITCache:
    def __init__(self, cache_dir = DEFAULT_CACHE_DIR) -> None:
        self.idx = 0 # unique id of saved compiled .so
        self.cache_dir = cache_dir
        self.cache = { IDX_KEY : self.idx } # key: ait op hash str, value: relative path of compiled .so
        self.fp = None
    
    def _open(self):
        if not os.path.exists(os.path.join(self.cache_dir, CACHE_FILE_NAME)):
            self.fp = open(os.path.join(self.cache_dir, CACHE_FILE_NAME), "w")
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
            self.cache = { IDX_KEY : self.idx }
            json.dump(self.cache, self.fp, indent=2)
            fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
            self.fp.close()
        self.fp = open(os.path.join(self.cache_dir, CACHE_FILE_NAME), "r+")
        # print("try to acquire file lock...")
        # acquire lock
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_EX)
        # print("file lock acquired.")

    def _close(self):
        # release lock
        fcntl.flock(self.fp.fileno(), fcntl.LOCK_UN)
        self.fp.close()
    
    def _load(self):
        try:
            self.cache = json.load(self.fp)
        except json.decoder.JSONDecodeError:
            self.cache = { IDX_KEY : self.idx }

    def _save(self):
        self.fp.seek(0)
        json.dump(self.cache, self.fp, indent=2)

    def sync_cache(self):
        # sync cache with the files in cache dir 
        # remove invalid key-value pairs
        max_idx = 0
        for gpu_type in self.cache:
            if gpu_type == IDX_KEY:
                continue
            for key, value in self.cache[gpu_type].items():
                if not os.path.exists(os.path.join(self.cache_dir, value)):
                    self.cache[gpu_type].pop(key)
                    continue
                if self.get_lib_idx(value) > max_idx:
                    max_idx = self.get_lib_idx(value)
        self.idx = max_idx + 1
        self.cache[IDX_KEY] = self.idx
    
    def load_or_create_cache(self):
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir, exist_ok=True)
        self._open()
        self._load()
        self.sync_cache()
    
    def add(self, gpu_type, key, lib_path, override = False):
        if gpu_type not in self.cache:
            self.cache[gpu_type] = {}
        if override or key not in self.cache[gpu_type]:
            value = "{:0>16}.so".format(self.idx) 
            self.idx += 1
            self.cache[IDX_KEY] = self.idx
            self.cache[gpu_type][key] = value
            copyfile(lib_path, os.path.join(self.cache_dir, value))
            copymode(lib_path, os.path.join(self.cache_dir, value))
    
    def find(self, gpu_type, key):
        if gpu_type not in self.cache:
            return None
        if key in self.cache[gpu_type]:
            return os.path.join(self.cache_dir, self.cache[gpu_type][key])
        else:
            return None

    def get_lib_idx(self, lib_name):
        return int(lib_name.split(".")[0])
    
    def close_cache(self):
        if self.fp != None:
            self._close()