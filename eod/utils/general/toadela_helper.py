import os
import time

from spring_aux.adela.adela import Adela

from .tokestrel_helper import to_kestrel
from eod.utils.general.cfg_helper import merge_opts_into_cfg
from eod.utils.general.log_helper import default_logger as logger


def to_adela(config, release_name, save_to=None, serialize=False):
    opts = config.get('args', {}).get('opts', [])
    config = merge_opts_into_cfg(opts, config)
    # to kestrel
    kestrel_save_path = to_kestrel(config, save_to, serialize)
    # generate release.json
    if config.get('adela', None):
        release_name = release_name
        cmd = 'python -m spring_aux.adela.make_json {} -o {}'.format(kestrel_save_path, release_name)
        os.system(cmd)
        # self.to_adela(release_name)

    cfg_adela = config.get('adela', None)
    assert cfg_adela is not None, 'need adela configuration.'
    # build adela
    adela = Adela(server=cfg_adela['server'])
    adela.setup()

    # add release id
    pid = cfg_adela['pid']
    release_rep = adela.release_add(pid=pid, fname=release_name)
    rid = release_rep.id
    logger.info(f'release id:{rid}')

    # deploy
    dep_params = cfg_adela['dep_params']
    # add quantity dataset
    if dep_params.get('dataset_added', None):
        adela.dataset_add(pid, dep_params['dataset_added'])
        logger.info(f'quantity dataset add.')
    dep_rep = adela.deployment_add_by_param(pid=pid, rid=rid, info=dep_params)
    did = dep_rep.id
    logger.info(f'deploy id:{did}')
    # wait for deployment response
    status = adela.deployment(pid, did).status
    while(status != "SUCCESS"):
        if status == "FAILURE":
            logger.warning('deploy failed')
            exit(0)
        time.sleep(1)
        status = adela.deployment(pid, did).status

    # benchmark
    precision_params = cfg_adela.get('precision_params', None)
    if precision_params:
        # add benchmark dataset
        if precision_params.get('dataset_added', None):
            adela.dataset_add(pid, precision_params['dataset_added'])
            logger.info(f'benchmark dataset add.')
        bid = adela.add_benchmark(pid, did, precision_params).id
        logger.info(f'benchmark id:{bid}')
        status = adela.benchmark(pid, did, bid)["status"]
        while(status != "SUCCESS"):
            if status == "FAILURE":
                logger.warning('benchmark failed')
                exit(0)
            time.sleep(1)
            status = adela.benchmark(pid, did, bid)["status"]
        # evaluate results display
        print(adela.benchmark(pid, did, bid))
