import torch
import math
import time
import random
import numpy as np
import torch.utils.checkpoint as checkpoint
from .log_helper import default_logger as logger
from up.utils.env.dist_helper import all_gather


def format_size(tensor_size):
    units = ['B', 'KB', 'MB', 'GB']
    for i in range(len(units) - 1):
        if tensor_size <= 1024:
            return f"{tensor_size:.2f} {units[i]}"
        tensor_size /= 1024
    return f"{tensor_size:.2f} {units[-1]}"


def store_rng_state():
    torch_rng_state = torch.get_rng_state()
    torch_cuda_rng_state = torch.cuda.get_rng_state()
    np_rng_state = np.random.get_state()
    rd_rng_state = random.getstate()

    return {
        "torch_rng_state": torch_rng_state,
        "torch_cuda_rng_state": torch_cuda_rng_state,
        "np_rng_state": np_rng_state,
        "rd_rng_state": rd_rng_state,
    }


def cast_checkpoint(func, *args, **kwargs):
    def create_custom_forward(module, **kwargs):
        def custom_forward(*inputs):
            return module(*inputs, **kwargs)
        return custom_forward

    return checkpoint.checkpoint(create_custom_forward(func, **kwargs), *args)


def common_cast_forward(m, *args, **kwargs):
    old_forward = m.forward

    def forward(*args, **kwargs):
        return cast_checkpoint(old_forward, *args, **kwargs)
    m.forward = forward
    m.old_forward = old_forward


def dc_cast_forward(module_m, name, manager):
    old_forward = module_m.forward

    def profile_function(func, *args, **kwargs):
        prev_allocated = torch.cuda.memory_allocated()
        torch.cuda.synchronize()
        prev_time = time.time()
        ret = func(*args, **kwargs)
        torch.cuda.synchronize()
        data = time.time() - prev_time, torch.cuda.memory_allocated() - prev_allocated
        return DataStorage(*data), ret

    def forward(*args, **kwargs):
        """ 共有四种情况 """
        """
            1. 当前层使用 checkpoint
            2. 上层使用 checkpoint
            3. 下层使用 checkpoint
            4. 下层未使用 checkpoint
        """
        if name in manager.cur_checkpoint_modules:
            if manager.is_warmup:
                rng_state = store_rng_state()
                """ warmup 阶段只会checkpoint max_levels // 2的block"""
                data, ret = profile_function(old_forward, *args, **kwargs)
                # checkpoint 会保留输出的 tensor
                data.mem_allocated[0] -= int(np.array(ret[0].shape).prod() * ret[0].element_size())
                manager.checkpoint_reduce_memory += data.mem_allocated[0]
                manager.add_data(name, data)
                restore_rng_state(**rng_state)
            manager.prev_use_checkpoint()
            ret = cast_checkpoint(old_forward, *args, **kwargs)
            manager.post_use_checkpoint()
            return ret
        elif manager.under_checkpoint:
            return old_forward(*args, **kwargs)
        else:
            return old_forward(*args, **kwargs)
            # checkpoint_times = manager.checkpoint_count
            # data, ret = profile_function(old_forward, *args, **kwargs)
            # if manager.checkpoint_count == checkpoint_times:
            #     manager.add_data(name, data)
            # return ret
    module_m.forward = forward
    module_m.old_forward = old_forward


class PolyPrediction(object):
    def __init__(self, x, y) -> None:
        super().__init__()
        self.poly_func = self.fit_poly(x, y)

    def check_fit(self, poly_func, x, y):
        y_val = poly_func(x)
        if isinstance(y, np.ndarray):
            y_np = y
        else:
            y_np = np.array(y)
        # at least 1 KB
        rel_error = np.max(np.abs(y_np - y_val) / np.stack([y_np, [1024] * y_np.size]).max(axis=0))
        # error < 10 MB
        fit_flag = rel_error < 0.1 or np.max(np.abs(y_np - y_val)) < 1e7
        if not fit_flag:
            logger.warning(f"rel error: {rel_error:.2%}, {np.max(np.abs(y_np - y_val)) / (1024 ** 2):0.2f} MB")
        return fit_flag

    def fit_poly(self, x, y, deg=2):
        # 当输入不变时，认为显存占用也不变
        if len(set(x)) == 1:
            deg = 0
        poly_param = np.polyfit(x, y, deg)
        poly_func = np.poly1d(poly_param)
        if not self.check_fit(poly_func, x, y):
            logger.warning("Memory consumption cannot be fitted to a quadratic polynomial")
        return poly_func

    def predict(self, *args, **kwargs):
        return self.poly_func(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def update(self, x, y, update_p=0.2, max_x=1600 * 1600 * 3, min_x=400 * 400 * 3):
        random_number = math.ceil(1 / update_p)
        random_number = max(random_number, 4)
        random_x = np.array(list(range(random_number))) * (max_x - min_x) / random_number + min_x
        random_y = self.predict(random_x)

        new_x = random_x.tolist()
        new_x.append(x)
        new_y = random_y.tolist()
        new_y.append(y)

        self.poly_func = self.fit_poly(new_x, new_y)


class DataStorage:
    """ Memory consumption and forward time for a specific input size """

    def __init__(self, time_use, mem_allocated):
        self.time_use = [time_use]
        self.mem_allocated = [mem_allocated]

    def add(self, data_storage):
        self.time_use += data_storage.time_use
        self.mem_allocated += data_storage.mem_allocated

    def get_time(self):
        return np.mean(self.time_use[:1])

    def get_memory(self):
        return np.mean(self.mem_allocated)

    def serialize(self):
        return {"time": self.time_use, "memory": self.mem_allocated}

    def __str__(self):
        return f"time: {self.get_time() * 1e3:.3f} ms, memory: {format_size(self.get_memory())}"


def restore_rng_state(torch_rng_state=None, torch_cuda_rng_state=None, np_rng_state=None, rd_rng_state=None):
    if torch_rng_state is not None:
        torch.set_rng_state(torch_rng_state)
    if torch_cuda_rng_state is not None:
        torch.cuda.set_rng_state(torch_cuda_rng_state)
    if np_rng_state is not None:
        np.random.set_state(np_rng_state)
    if rd_rng_state is not None:
        random.setstate(rd_rng_state)


class BaseStrategy:
    def __init__(self) -> None:
        pass

    def get_checkpoint_module(self, *args, **kwargs):
        raise NotImplementedError("You must implement \"get_checkpoint_module\" method")


class GreedyStrategy(BaseStrategy):
    def __init__(self, module_func, checkpoint_module=None) -> None:
        super().__init__()
        self.memory_predict_func = module_func
        self.checkpoint_module = checkpoint_module

    def sort_by_name(self, bucket):
        # TODO:可以在第一个iter里面设置开始结束时间，以此来确定前后顺序
        # 这里的顺序是后面的module在前面，优先级从低到高
        tag_bucket = []
        for value in bucket:
            layer_index = value[-1].split('.')[-1]
            if layer_index.isdigit():
                layer_index = int(layer_index)
            else:
                layer_index = 0
            tag_bucket.append((layer_index, value))
        tag_bucket.sort(reverse=True)
        return [value for _, value in tag_bucket]

    def split_bucket(self, module_memory: list((float, str))):
        # 显存占用排序，从大到小
        module_memory.sort(reverse=True)
        new_module_memory = []
        i = 0
        while i < len(module_memory):
            # 设置分桶的分界点
            memory_threshold = module_memory[i][0] * 0.9
            bucket = [module_memory[i]]
            i += 1
            while i < len(module_memory):
                if module_memory[i][0] >= memory_threshold:
                    bucket.append(module_memory[i])
                    i += 1
                else:
                    break
            new_module_memory += self.sort_by_name(bucket)
        # 优先级从低到高
        return new_module_memory

    def get_checkpoint_module(self, input_size, reduce_memory):
        module_memory = []
        max_activation = 0
        for name in self.checkpoint_module:
            func = self.memory_predict_func[name]
            memory_size = func(input_size)
            max_activation = max(max_activation, memory_size)
            module_memory.append((memory_size, name))
        module_memory = self.split_bucket(module_memory)

        module_set = set()
        # TODO: 需要判断最后一层是否使用checkpoint
        # reduce_memory += max_activation
        if reduce_memory <= 0:
            return module_set

        def get_fit_module(module_memory, target):
            for value in module_memory[::-1]:
                if value[0] >= target:
                    return value
            return module_memory[-1]

        curr_memory = 0
        candidate_modules = module_memory.copy()
        while curr_memory < reduce_memory and len(candidate_modules) > 0:
            entry = get_fit_module(candidate_modules, reduce_memory - curr_memory)
            curr_memory += entry[0]
            module_set.add(entry[1])
            candidate_modules.remove(entry)

        return module_set


class BagGreedyStrategy(BaseStrategy):
    def __init__(self, memory_func, time_func, checkpoint_module=None, bin_size=10):
        super().__init__()
        self.memory_predict_func = memory_func
        self.time_predict_func = time_func
        self.checkpoint_module = checkpoint_module
        self.bin_size = bin_size

    def get_predict_memory_time(self, input_size):
        module_memory = []
        module_time = []
        for name in self.checkpoint_module:
            if name not in self.memory_predict_func:
                continue
            memory_size = self.memory_predict_func[name](input_size)
            module_memory.append(memory_size)
            module_time.append((self.time_predict_func[name](input_size), memory_size, name))
        return module_time

    def get_bag_solution(self, module_time, target_memory):
        '''
        module_time: (time, memory, name)
        '''
        target_modules = []
        dp = []
        N = len(module_time)
        M = math.ceil(target_memory / self.bin_size / 1024 / 1024) * 2
        module_time.sort()
        time_memory_name = [(0, 0, '')]
        for item in module_time:
            m_size = math.ceil(max(0, item[1]) / self.bin_size / 1024 / 1024)
            time_memory_name.append((item[0], m_size, item[2]))
        dp = np.ones((N + 1, M + 1)) * -10000
        path = np.zeros((N + 1, M + 1))
        for i in range(1, M + 1):
            if i >= time_memory_name[1][1]:
                dp[0][i] = time_memory_name[1][0]
        for i in range(1, N + 1):
            for j in range(1, M + 1):
                if j < time_memory_name[i][1]:
                    dp[i][j] = dp[i - 1][j]
                else:
                    x = dp[i - 1][j]
                    y = dp[i - 1][j - time_memory_name[i][1]] + time_memory_name[i][0]
                    if x < y:
                        dp[i][j] = x
                        path[i][j] = 0
                    else:
                        dp[i][j] = y
                        path[i][j] = 1

        def find_path(n, m):
            memory = 0
            module_list = []
            while n > 0:
                if path[n][m] > 0:
                    m -= time_memory_name[n][1]
                    memory += time_memory_name[n][1]
                    module_list.append(time_memory_name[n][2])
                n -= 1
                if m < 0:
                    break
            return memory, module_list
        for i in range(M // 2, M):
            memory, module_list = find_path(N, i)
            if memory > target_memory / self.bin_size / 1024 / 1024:
                target_modules = module_list
                break
        if len(target_modules) == 0:
            target_modules = self.checkpoint_module
        return target_modules

    def get_checkpoint_module(self, input_size, reduce_memory):
        module_time = self.get_predict_memory_time(input_size)
        return self.get_bag_solution(module_time, reduce_memory)


class DynamicCheckpointManager(object):
    def __init__(self,
                 warmup_iters=30,
                 checkpoint_modules=[],
                 max_memory=8,
                 debug_freq=-1,
                 strategy="greedy",
                 share_weight_node={}):
        self.warmup_iters = warmup_iters
        self.all_checkpoint_modules = checkpoint_modules
        self.cur_checkpoint_modules = checkpoint_modules
        self.size_checkpoint_modules = {}
        self.cached_strategy = {}
        self.model_memory_data = {}
        self.model_memory_predict = None
        self.input_size = 0
        self.checkpoint_count = 0
        self.iters = 0
        self.checkpoint_size_data = {}
        self.memory_predict_func = {}
        self.time_predict_func = {}
        self.interval = 100
        self.max_memory = max_memory * (1024 ** 3)
        self.under_checkpoint = False
        self.debug_freq = debug_freq
        self.strategy_name = strategy
        self.share_weight_node = share_weight_node

    def set_max_memory_GB(self, memory_threshold):
        self.max_memory = memory_threshold * (1024 ** 3)

    def round_input(self, input_size):
        round_input_size = math.ceil(input_size / self.interval) * self.interval
        return round_input_size

    def init_dc_strategy(self):
        if self.strategy_name == 'memory_time':
            self.strategy = BagGreedyStrategy(self.memory_predict_func,
                                              self.time_predict_func, self.all_checkpoint_modules)
        elif self.strategy_name == 'greedy':
            self.strategy = GreedyStrategy(self.memory_predict_func, self.all_checkpoint_modules)
        else:
            logger.info(f"{self.strategy_name} is not supported")

    def before_forward(self):
        self.iters += 1
        self.checkpoint_reduce_memory = -torch.cuda.memory_allocated()

        if self.iters == self.warmup_iters + 1:
            self.fit_memory_consume()
            self.init_dc_strategy()

        self.checkpoint_count = 0
        self.cur_checkpoint_modules, reduce_memory = self.get_current_checkpoint_modules()
        if self.input_size not in self.checkpoint_size_data:
            self.checkpoint_size_data[self.input_size] = {}

        if self.debug_freq > 0:
            if self.iters % self.debug_freq == 0 and not self.is_warmup:
                self.debug_info(reduce_memory)

        if self.is_warmup:
            torch.cuda.empty_cache()
            torch.cuda.memory.reset_peak_memory_stats()

    @property
    def is_warmup(self):
        return self.iters <= self.warmup_iters

    def debug_info(self, reduce_memory=0):
        predict_model_memory = self.model_memory_predict(self.input_size) / (1024 ** 2)
        round_input_size = self.round_input(self.input_size)
        predict_model_round_memory = self.model_memory_predict(round_input_size) / (1024 ** 2)
        logger.info("================= schedule begin =================")
        input_size = np.sqrt(self.input_size)
        logger.info(f"input size (sqrt): {input_size:.2f}")
        logger.info(f"memory threshold: {self.max_memory / (1024 ** 2):.2f} MB")
        logger.info(f"current memory usage: {torch.cuda.memory_allocated() / (1024 ** 2):.2f} MB")
        logger.info(f"predict model memory: {predict_model_memory:.2f} MB")
        logger.info(f"predict model memory(round input): {predict_model_round_memory:.2f} MB")
        logger.info(f"reduce memory: {reduce_memory / (1024 ** 2):.2f} MB")
        logger.info("checkpoint modules:")
        for module_id in self.cur_checkpoint_modules:
            memory_saved = self.memory_predict_func[module_id](self.input_size) / (1024 ** 2)
            logger.info(f"\t{module_id}: {memory_saved:.2f} MB")

        logger.info("================== schedule end ==================")

    def get_current_checkpoint_modules(self):
        local_checkpoint_module = set()
        reduce_memory = 0
        if self.is_warmup:
            local_checkpoint_module = self.all_checkpoint_modules
        else:
            round_input_size = self.round_input(self.input_size)
            if round_input_size in self.cached_strategy:
                local_checkpoint_module, reduce_memory = self.cached_strategy[round_input_size]
            else:
                available_memory = self.max_memory - torch.cuda.memory_allocated()
                reduce_memory = self.model_memory_predict(round_input_size) - available_memory
                if reduce_memory > 0:
                    local_checkpoint_module = self.strategy.get_checkpoint_module(round_input_size, reduce_memory)
                self.cached_strategy[round_input_size] = (local_checkpoint_module, reduce_memory)
        return local_checkpoint_module, reduce_memory

    def prev_use_checkpoint(self):
        self.under_checkpoint = True
        self.checkpoint_count += 1

    def post_use_checkpoint(self):
        self.under_checkpoint = False

    def add_data(self, name, data):
        if name in self.checkpoint_size_data[self.input_size]:
            self.checkpoint_size_data[self.input_size][name].add(data)
        else:
            self.checkpoint_size_data[self.input_size][name] = data

    def collect_model_memory(self, memory):
        if self.input_size not in self.model_memory_data:
            self.model_memory_data[self.input_size] = DataStorage(time_use=0, mem_allocated=memory)
        else:
            self.model_memory_data[self.input_size].add(DataStorage(time_use=0, mem_allocated=memory))

    def after_update(self):
        if self.is_warmup and self.iters >= 1:
            self.collect_model_memory(torch.cuda.max_memory_allocated() + self.checkpoint_reduce_memory)

    def fit_memory_consume(self):
        name2data = {}  # {"module" : {"x": [], "y": []}}
        # collect data

        def get_data(data, name):
            if name not in self.share_weight_node:
                return data
            else:
                new_data = []
                num = self.share_weight_node[name]
                n = len(data) // num
                for i in range(n):
                    sum = 0
                    for j in range(num):
                        sum += data[i * num + j]
                    new_data.append(sum)
                return new_data

        for input_size, data_map in self.checkpoint_size_data.items():
            for name, data_storage in data_map.items():
                if name not in name2data:
                    name2data[name] = {"input_size": [], "memory": [], "time": []}
                m_data = get_data(data_storage.mem_allocated, name)
                t_data = get_data(data_storage.time_use, name)
                name2data[name]["input_size"] += [input_size] * len(m_data)
                name2data[name]["memory"] += m_data
                name2data[name]['time'] += t_data

        # all gather
        for name, value in name2data.items():
            input_size = value['input_size'][1:]
            memory = value['memory'][1:]
            time = value['time'][1:]
            value['input_size'] = np.array(all_gather(input_size)).flatten().tolist()
            value['memory'] = np.array(all_gather(memory)).flatten().tolist()
            value['time'] = np.array(all_gather(time)).flatten().tolist()
        # fit
        for name, value in name2data.items():
            self.memory_predict_func[name] = PolyPrediction(value["input_size"], value["memory"])
            m_func = self.memory_predict_func[name]
            if not m_func.check_fit(m_func.poly_func, value["input_size"], value["memory"]):
                logger.warning(f"{name} dose not fit ploy memory predict")
            self.time_predict_func[name] = PolyPrediction(value["input_size"], value['time'])
            t_func = self.time_predict_func[name]
            if not t_func.check_fit(t_func.poly_func, value["input_size"], value["time"]):
                logger.warning(f"{name} dose not fit ploy time predict")
        # do this for whole model
        input_size_list = []
        memory_list = []
        for input_size, data_storage in self.model_memory_data.items():
            input_size_list += [input_size] * len(data_storage.mem_allocated)
            memory_list += data_storage.mem_allocated

        # all gather
        input_size_list = np.array(all_gather(input_size_list)).flatten().tolist()
        memory_list = np.array(all_gather(memory_list)).flatten().tolist()

        self.model_memory_predict = PolyPrediction(input_size_list, memory_list)
        logger.info("fit memory consume")
