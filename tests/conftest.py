import os
import sys


# Windows에서 pytest.exe로 실행해도 저장소 루트의 src 패키지를 찾게 한다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
