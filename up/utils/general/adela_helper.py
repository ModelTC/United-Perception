from __future__ import division
# Import from up
from up.utils.general.log_helper import default_logger as logger
from up.utils.general.registry_factory import TOADELA_REGISTRY


__all__ = ['BaseToAdela']


@TOADELA_REGISTRY.register('base')
class BaseToAdela(object):
    def __init__(self, cfg_adela, save_to):
        self.cfg_adela = cfg_adela
        self.tar_model = save_to

    def process(self):
        from spring_aux.adela.adela import Adela
        import time

        cfg_adela = self.cfg_adela
        tar_model = self.tar_model
        # build adela
        adela = Adela(server=cfg_adela['server'], auth="~/.adela.config")
        adela.setup()

        pid = cfg_adela['pid']
        # release
        release_rep = adela.release_add_tar(pid=pid, fname=tar_model, train_info="{}")
        rid = release_rep.id
        logger.info(f'release id:{rid}')

        # deploy & quantity
        dep_params = cfg_adela['dep_params']
        dids = []
        for dep_p in dep_params:
            # add quantity dataset
            if dep_p.get('dataset_added', None):
                adela.dataset_add(pid, dep_p['dataset_added'])
                logger.info(f'quantity dataset add.')
            dep_rep = adela.deployment_add_by_param(pid=pid, rid=rid, info=dep_p)
            did = dep_rep.id
            dids.append(did)
            logger.info(f'deploy id:{did}')
            # wait for deployment response
            status = adela.deployment(pid, did).status
            while (status != "SUCCESS"):
                if status == "FAILURE":
                    logger.warning('deploy failed')
                    exit(0)
                time.sleep(1)
                status = adela.deployment(pid, did).status

        # benchmark
        precision_params = cfg_adela.get('precision_params', None)
        if precision_params:
            didxs = precision_params.get('didxs', [_ for _ in range(len(dids))])
            prec_params = precision_params['kwargs']
            if not isinstance(prec_params, list):
                prec_params = [prec_params]
            assert len(didxs) == len(prec_params), "make sure didxs length equal to prec_params."
            for idx in range(len(prec_params)):
                prec_p, didx = prec_params[idx], didxs[idx]
                # add benchmark dataset
                if precision_params.get('dataset_added', None):
                    adela.dataset_add(pid, prec_p['dataset_added'])
                    logger.info(f'benchmark dataset add.')
                bid = adela.add_benchmark(pid, dids[didx], prec_p).id
                logger.info(f'benchmark id:{bid}')
                status = adela.benchmark(pid, dids[didx], bid)["status"]
                while (status != "SUCCESS"):
                    if status == "FAILURE":
                        logger.warning('benchmark failed')
                        exit(0)
                    time.sleep(1)
                    status = adela.benchmark(pid, dids[didx], bid)["status"]

        # download
        publish_params = cfg_adela.get('publish_params', None)
        if publish_params:
            didxs = publish_params.get('didxs', [_ for _ in range(len(dids))])
            for didx in didxs:
                mid = adela.deployment(pid, dids[didx]).model_id
                logger.info(f'model id:{mid}')
                # publish
                if mid == -1:
                    mid = adela.model_add(pid, dids[didx]).id
                adela.model_download(pid, mid)
        logger.info(f'adela done')
