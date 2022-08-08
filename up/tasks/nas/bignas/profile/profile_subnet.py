import time
import json
import sys

# from spring.models.latency import Latency
from spring.models.latency import Latency
from up.utils.general.log_helper import default_logger


def test_latency(model_file,
                 hardware_name='T4', backend_name='cuda11.0-trt7.1',
                 data_type='int8', batch_size=8,
                 graph_name='test',
                 match_mode=1, match_speed_mode=1, match_speed_async=False):
    latency_client = Latency()
    ret = latency_client.call(
        hardware_name,
        backend_name,
        data_type,
        batch_size,
        model_file,
        graph_name=graph_name,
        force_test=False,
        print_info=False,
        match_mode=match_mode,
        match_speed_mode=match_speed_mode,
        match_speed_async=match_speed_async
    )
    default_logger.info(json.dumps(ret))
    if not ret or 'ret' not in ret.keys() or 'status' not in ret['ret'].keys() or \
            ret['ret']['status'] != 'success':
        default_logger.info('test latency failed')
        cnt = 0
        while not ret or 'ret' not in ret.keys() or 'status' not in ret['ret'].keys() or \
                ret['ret']['status'] != 'success':
            time.sleep(1)
            ret = latency_client.call(
                hardware_name,
                backend_name,
                data_type,
                batch_size,
                model_file,
                graph_name=graph_name,
                force_test=False,
                print_info=False,
                match_mode=match_mode,
                match_speed_mode=match_speed_mode,
                match_speed_async=True
            )
            cnt += 1
    return ret['cost_time']


if __name__ == '__main__':
    onnx_path = sys.argv[1]
    test_latency(onnx_path)
