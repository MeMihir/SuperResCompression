import os
def comp_ratio(path1, path2):
  return os.path.getsize(path1)/os.path.getsize(path2)