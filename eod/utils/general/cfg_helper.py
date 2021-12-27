# Standard Library
import copy
import json
import re

from .log_helper import default_logger as logger


def format_cfg(cfg):
    """Format experiment config for friendly display"""
    # json_str = json.dumps(cfg, indent=2, ensure_ascii=False)
    # return json_str

    def list2str(cfg):
        for key, value in cfg.items():
            if isinstance(value, dict):
                cfg[key] = list2str(value)
            elif isinstance(value, list):
                if len(value) == 0 or isinstance(value[0], (int, float)):
                    cfg[key] = str(value)
                else:
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            value[i] = list2str(item)
                    cfg[key] = value
        return cfg

    cfg = list2str(copy.deepcopy(cfg))
    json_str = json.dumps(cfg, indent=2, ensure_ascii=False).split("\n")
    # json_str = [re.sub(r"(\"|,$|\{|\}|\[$|\s$)", "", line) for line in json_str if line.strip() not in "{}[]"]
    json_str = [re.sub(r"(\"|(!\],$)|\s$)", "", line) for line in json_str]
    cfg_str = "\n".join([line.rstrip() for line in json_str if line.strip()])
    return cfg_str


def is_number(num):
    pattern = re.compile(r'^[-+]?[-0-9]\d*\.\d*|[-+]?\.?[0-9]\d*$')
    res = pattern.match(num)
    if res:
        return True
    return False


def try_decode(val):
    """bool, int, float, or str"""
    if val.upper() == 'FALSE':
        return False
    elif val.upper() == 'TRUE':
        return True
    if val.isdigit():
        return int(val)
    if is_number(val):
        return float(val)
    return val


def merge_opts_into_cfg(opts, cfg):
    cfg = copy.deepcopy(cfg)
    if opts is None or len(opts) == 0:
        return cfg
    
    assert len(opts) % 2 == 0
    keys, values = opts[0::2], opts[1::2]
    for key, val in zip(keys, values):
        logger.info(f'replacing {key}')
        val = try_decode(val)
        cur_cfg = cfg
        # for hooks
        if '-' in key:
            key_p, key_s = key.split('-')
            k_module, k_type = key_p.split('.')
            cur_cfg = cur_cfg[k_module]
            flag_exist = False
            for idx in range(len(cur_cfg)):
                if cur_cfg[idx]['type'] != k_type:
                    continue
                flag_exist = True
                cur_cfg_temp = cur_cfg[idx]
                key_s = key_s.split('.')
                for k in key_s[:-1]:
                    cur_cfg_temp = cur_cfg_temp.setdefault(k, {})
                cur_cfg_temp[key_s[-1]] = val
            if not flag_exist:
                _cur_cfg = {}
                cur_cfg_temp = _cur_cfg
                key_s = key_s.split('.')
                for k in key_s[:-1]:
                    cur_cfg_temp = cur_cfg_temp.setdefault(k, {})
                cur_cfg_temp[key_s[-1]] = val
                cur_cfg.append(_cur_cfg)
        else:
            key = key.split('.')
            for k in key[:-1]:
                cur_cfg = cur_cfg.setdefault(k, {})
            cur_cfg[key[-1]] = val
    return cfg


def upgrade_cfg(cfg):
    # cfg = upgrade_fp16(cfg)
    return cfg
