"""
Инициализация модуля datasets с автоматической настройкой PYTHONPATH.
Обеспечивает доступ к корневым модулям проекта из любых скриптов подкаталога.
"""

import os
import sys


# Добавляем корневую директорию проекта в PYTHONPATH
_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

# Предварительно импортируем часто используемые модули
try:
    from entrypoint.config import BASE_DIR
    __all__ = ['BASE_DIR']
except ImportError:
    pass