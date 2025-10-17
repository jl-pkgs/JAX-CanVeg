import time
import functools
from typing import Callable, Any

def timer(ntime: int = 5) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """带参数的计时装饰器。

    使用方式：
      @timer(ntime=10)
      def foo(...): ...
    """
    def _decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(func)
        def _wrapper(*args, **kwargs) -> Any:
            start = time.perf_counter()

            result = func(*args, **kwargs)
            for _ in range(ntime - 1):
                result = func(*args, **kwargs)

            elapsed = time.perf_counter() - start
            avg = elapsed / max(ntime, 1)
            print(f"⏱️ '{func.__name__}' {ntime} times: sum {elapsed:.4f}s, avg: {avg:.4f}s")
            return result

        return _wrapper

    return _decorator


if __name__ == "__main__":

    @timer()
    def slow_function(n):
        """一个慢函数"""
        total = 0
        for i in range(n):
            total += i ** 2
        return total

    # 测试
    result = slow_function(1000000)
    print(f"结果: {result}")
