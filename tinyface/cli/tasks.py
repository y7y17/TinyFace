from queue import Queue
from typing import Any, Callable, List, Optional, Tuple

from async_tasks.executor import AsyncTasksExecutor
from tqdm import tqdm


def run_tasks(
    tasks: List[Callable],
    desc: str = "Processing",
    max_workers: Optional[int] = None,
):
    if not tasks:
        return []

    results: List[Tuple[Optional[Exception], Any]] = []
    result_queue: Queue[Tuple[Any, Optional[Exception]]] = Queue()

    if max_workers is None:
        max_workers = min(6, len(tasks))

    def process_task(task: Callable) -> None:
        try:
            res = task()
            result_queue.put((None, res))
        except Exception as err:
            result_queue.put((err, None))

    with AsyncTasksExecutor(max_workers=max_workers) as executor:
        for task in tasks:
            executor.submit(process_task, task)

        completed = 0
        with tqdm(total=len(tasks), desc=desc) as process:
            try:
                while completed < len(tasks):
                    results.append(result_queue.get())
                    completed += 1
                    process.update(1)
            except KeyboardInterrupt:
                executor.shutdown()

    return results
