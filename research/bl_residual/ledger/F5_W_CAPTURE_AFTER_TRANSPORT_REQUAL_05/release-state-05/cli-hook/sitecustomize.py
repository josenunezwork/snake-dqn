import os,runpy
try:
    runpy.run_path('/Users/josenunez/Projects/ml/snake-dqn-f5-cap05/research/colab_cli/cli_guard.py', run_name='snake_cli_guard')
except BaseException:
    os._exit(70)
