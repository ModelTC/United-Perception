from easydict import EasyDict
from .distiller import FeatureMimicFeature
from eod.utils.general.log_helper import default_logger as logger


__all__ = ["MimicJob", "Mimicker"]


class MimicJob(object):
    """
    A basic class providing abstract for a mimic job.
    """
    def __init__(self, mimic_name, mimic_type, s_name, t_name, mimic_loss_weight=1.0, **kwargs):
        """Init a mimic job.

        Args:
            mimic_name: str. The name given to the mimic job. DO NOT register multiple jobs sharing a same name.
            mimic_type: str. Losses provided in distiller. E.g, "l2" or "kd".
            s_name: list of str. A list of layer names in student model that need to be used for mimic.
                    E,g, ['neck.p2_conv', 'neck.p6_pool']
            t_name: list of str. A list of layer names in teacher model that need to be used for mimic.
                    E,g, ['neck.p2_conv', 'neck.p6_pool']
                    Note that names with same index in s_name and t_name are matched as a pair in mimicking, thus
                    the length of s_name is equal to the length of t_name.
            mimic_loss_weight: float. Mimic job loss weight.
            kwargs: dict. Other args that are needed to build the loss. E.g, you can specify "normalize" in l2
            loss. See losses in the distiller for more details.
        """
        logger.info("name: spring.distiller.mimicjob, mimic_name : {}, mimic_type: {}".format(mimic_name, mimic_type))
        assert isinstance(s_name, list), 's_name should be a list object.'
        assert isinstance(t_name, list), 't_name should be a list object.'
        self.mimic_name = mimic_name
        self.mimic_type = mimic_type
        self.mimic_loss_weight = mimic_loss_weight
        self.s_name = s_name
        self.t_name = t_name
        self.kwargs = kwargs

    @property
    def config(self):
        """Return a dict of mimicjob's configuration."""
        cfg_dict = {'type': self.mimic_type,
                    'name': self.mimic_name,
                    'kwargs': {
                        'student_layer_name': self.s_name,
                        'teacher_layer_name': self.t_name,
                        'weight': self.mimic_loss_weight
                    }
                    }
        cfg_dict['kwargs'].update(self.kwargs)
        return EasyDict(cfg_dict)


class Mimicker(object):
    def __init__(self, teacher_model=None, student_model=None):
        """
        Args:
            teacher_model: Teacher model.
            student_model: Student model.
        """
        logger.info("name: spring.distiller.mimicker")
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.job_pool = []
        self.registered_job_name = set()
        self.mimicker = FeatureMimicFeature(task_helper=None)

        if not isinstance(self.teacher_model, list):
            self.teacher_model = [self.teacher_model]
        self.output_t_maps = [dict() for _ in self.teacher_model]
        self.output_s_maps = {}
        self.mimicker.adjust_layer = self._deprecate_adjustlayer

    def register_job(self, mimic_job):
        """Register a mimic job to the mimicker."""
        self._registry_sanity_check(mimic_job)
        self.job_pool.append(mimic_job)
        self.registered_job_name.add(mimic_job.mimic_name)

    def _registry_sanity_check(self, mimic_job):
        """Sanity check when registring a mimic job."""
        assert isinstance(mimic_job, MimicJob), 'Only MimicJob object can be registered.'
        assert mimic_job.mimic_name not in self.registered_job_name, 'job name \'{}\' is duplicated.'
        if len(self.teacher_model) == 1:
            for single_t_name in mimic_job.t_name:
                assert isinstance(single_t_name, str), 'type of mimic job {}\'s \
                t_name should be str.'.format(mimic_job.mimic_name)
        else:
            for single_t_name in mimic_job.t_name:
                assert isinstance(single_t_name, list) and len(single_t_name) == len(self.teacher_model), \
                    'each t_name of mimic job should have length of {}'.format(len(self.teacher_model))

    def _register_forward_hooks(self):
        """Register forward hook to obtain output features of the objective layer."""
        for mimic_job in self.job_pool:
            for _name in mimic_job.t_name:
                if isinstance(_name, list):
                    for t_idx, _single_name in enumerate(_name):
                        self.mimicker._register_hooks(self.teacher_model[t_idx],
                                                      _single_name,
                                                      self.output_t_maps[t_idx])
                else:
                    self.mimicker._register_hooks(self.teacher_model[0], _name, self.output_t_maps[0])
            for _name in mimic_job.s_name:
                self.mimicker._register_hooks(self.student_model, _name, self.output_s_maps)

    def mimic(self, **kwargs):
        """Excute all of the registered mimicjobs.
        Returns:
            mimic_losses: List. A list of losses of registered mimic jobs.
        """
        mimic_losses = []
        # generate mimic loss
        for mimic_job in self.job_pool:
            feature_t, feature_s = self._resolve_features(mimic_job)
            mimic_loss, _, _, _ = self.mimicker.mimic(feature_s, feature_t, mimic_job.config, **kwargs)
            mimic_losses.append(mimic_loss)  # loss weight has been multiplied in mimicker.mimic
        return mimic_losses

    def _resolve_features(self, mimic_job):
        feature_t = []
        feature_s = []
        for _name in mimic_job.t_name:
            if isinstance(_name, list):
                valid_ensemble_num = 0
                ensemble_t_feats = 0
                for t_idx, _single_name in enumerate(_name):
                    if _single_name is not None:
                        ensemble_t_feats += self.output_t_maps[t_idx][_single_name]
                        valid_ensemble_num += 1
                feature_t.append(ensemble_t_feats / valid_ensemble_num)
            else:
                feature_t.append(self.output_t_maps[0][_name])
        for _name in mimic_job.s_name:
            feature_s.append(self.output_s_maps[_name])
        return feature_t, feature_s

    def prepare(self):
        """Pre-work of each forward step."""
        self.output_t_maps = [dict() for _ in self.teacher_model]
        self.output_s_maps = {}
        self.mimicker.clear()
        self._register_forward_hooks()

    def get_output_maps(self):
        """Get output feature of layers that are registered in Mimicker, both in teacher and student models.
        Returns:
            teacher output features: A dict of teacher features used for mimic, if there are multiple teachers,
            then return a list of dicts.
            student output features: A dict of student features used for mimic.
        """
        assert isinstance(self.output_t_maps, list)
        # For compatibility
        if len(self.output_t_maps) == 1:
            return self.output_t_maps[0], self.output_s_maps
        else:
            return self.output_t_maps, self.output_s_maps

    @staticmethod
    def _deprecate_adjustlayer(s_feats, t_feats):
        """Here we deprecate adjust layer and provide a `get_output_maps` API to facilitate flexible and
           user-defined adjustment demand.
        """
        return s_feats, t_feats
