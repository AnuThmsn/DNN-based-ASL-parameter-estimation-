import os
import sys

print("Running phase 3 and 4...")
ret = os.system("python run_experiments.py")
if ret != 0:
    sys.exit(ret)

print("Running phase 5 to 8...")
ret = os.system("python run_experiments_2.py")
if ret != 0:
    sys.exit(ret)

print("Running final...")
ret = os.system("python run_final.py")
if ret != 0:
    sys.exit(ret)

print("All done!")

