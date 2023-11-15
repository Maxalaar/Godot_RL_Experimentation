import os
import time
import webbrowser
from multiprocessing import Process
from typing import List
import ray


def run_tensorboard():
    os.system('tensorboard --logdir ./ray_results/')


def list_to_url(list_monitoring: List[str]) -> str:
    url: str = ''
    url += 'http://localhost:6006/?pinnedCards=['

    for element in list_monitoring:
        url += '{"plugin"%3A"scalars"%2C"tag"%3A"'
        url += element.replace('/', '%2F')
        url += '"}'
        url += '%2C'

    url = url[:-3]
    url += ']&darkMode=true#timeseries'

    url = url.replace('[', '%5B')
    url = url.replace(']', '%5D')
    url = url.replace('{', '%7B')
    url = url.replace('}', '%7D')
    url = url.replace('"', '%22')

    return url


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init()

    list_learning = [
        'ray/tune/evaluation/episode_reward_mean',
        'ray/tune/evaluation/episode_len_mean',
        'ray/tune/episode_reward_mean',
        'ray/tune/info/learner/policy0/learner_stats/vf_explained_var',
    ]

    list_simulation = [
        'ray/tune/sampler_perf/mean_env_render_ms',
        'ray/tune/sampler_perf/mean_action_processing_ms',
        'ray/tune/sampler_perf/mean_inference_ms',
        'ray/tune/perf/cpu_util_percent',
        'ray/tune/perf/gpu_util_percent0',
        'ray/tune/perf/ram_util_percent',
    ]

    process: Process = Process(target=run_tensorboard)
    process.start()
    time.sleep(5)

    # Monitoring simulation
    webbrowser.open(list_to_url(list_simulation))

    # Monitoring of learning
    webbrowser.open(list_to_url(list_learning))

    # Monitoring Ray Dashboard
    webbrowser.open('127.0.0.1:8265')
