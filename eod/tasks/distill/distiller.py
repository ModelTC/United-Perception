import torch
from .losses import *
from .adjust_helper import AdjustFeatures
# from spring.nart.tools.io import send
import copy
# import threading

__all__ = ["ModelMimicModel", "ModelMimicFeature", "FeatureMimicFeature"]


class BaseDistiller(object):
    """Virtual basis class for distiller.

    BaseDistiller is a basis class and provides necessary util functions. Some functions are
    not implemented here and require implementation in child classes. Naming conventions follow
    rules below:
        s_*: student_ variable about student model (model)
        t_*: teacher_ variable about teacher model (model_teacher)
    """

    def __init__(self, task_helper=None):
        """Init BaseDistiller by registering loss functions"""
        self.loss_function_map = {}  # loss functions w.r.t. mimic type
        self._register_loss_function_map()

        self.handles = []
        self.s_output_map = {}
        self.t_output_map = {}
        self.mimic_job_loss = {}
        self.adjust_layer = None
        self.task_helper = task_helper

    def _register_input(self, **kwargs):
        """ Register inputs.

        Register input variables whose lifetime is within mimic() function. They are used in
        other inheritant functions, such as _config_check, _get_model_output. Do remember
        remove all registered input in _remove_intput
        """
        raise NotImplementedError

    def _remove_input(self):
        """Remove registered inputs.

        Remove registered variables at the end of mimic() function. It is called by clear(), and
        therefore it must be implemented.
        """
        raise NotImplementedError

    def _config_check(self, mimic_config):
        """Check mimic config.

        It performs sanity check on mimic config. mimic_type is guaranteed to exist in mimic config.
        It usually perform check on kwargs and features / layer names length. It is called by
        _sanity_check() function()
        """
        raise NotImplementedError

    def _build_output_map(self, mimic_config):
        """Register output hooks if necessary.

        Register output hooks on model according to mimic_config. It should consider all the student
        layer names and teacher layer names, if provided.
        """
        raise NotImplementedError

    def _get_model_output(self):
        """Run model to get output.

        Perform model / model_teacher forward with given input. Should register output in output_map with
        key = '$$$$s_output' or key = '$$$$t_output', which will be used by 'kd' mimic_type. It is called
        by _forward_model() function.

        Returns:
            s_output: output by student model
            t_output: output by teacher model
        """
        raise NotImplementedError

    def _build_distill_features(self, mimic_id, mimic_job):
        """Build features for distillation loss function.

        Translate layer_names to features or get features by using index on feature_groups. It is called by
        _knowledge_distill_loss() function.

        Returns:
            s_features: list of features extracted from student model
            t_features: list of features extracted from teacher model
        """
        return NotImplementedError

    def _register_loss_function_map(self):
        """Register mimic loss functions."""
        def _abloss(s_features, t_features, weight, **kwargs):
            """Activation Boundary. htttps://arxiv.org/pdf/1811.03233.pdf"""
            criterion_kd = ABLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _atloss(s_features, t_features, weight, **kwargs):
            """Attention loss. https://arxiv.org/abs/1612.03928"""
            criterion_kd = ATLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _ccloss(s_features, t_features, weight, **kwargs):
            """Correlation Congruence. https://arxiv.org/pdf/1904.01802.pdf"""
            criterion_kd = CCLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _cdloss(s_features, t_features, weight, **kwargs):
            """Channel Distillation: Channel-Wise Attention for Knowledge Distillation"""
            criterion_kd = CDLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _ftloss(s_features, t_features, weight, **kwargs):
            """Factor Transfor. https://arxiv.org/pdf/1802.04977.pdf"""
            criterion_kd = FTLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _jsloss(s_features, t_features, weight, **kwargs):
            """"Jensen–Shannon(JS) divergence loss."""
            criterion_kd = JSLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _kdloss(s_features, t_features, weight, **kwargs):
            """Knowledge distill loss. https://arxiv.org/abs/1503.02531"""
            criterion_kd = KDLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _celoss(s_features, t_features, weight, **kwargs):
            """Knowledge distill loss. https://arxiv.org/abs/1503.02531"""
            criterion_kd = CELoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _klloss(s_features, t_features, weight, **kwargs):
            """Kullback–Leibler(KL) divergence loss."""
            criterion_kd = KLLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _l2loss(s_features, t_features, weight, **kwargs):
            """L2 loss"""
            criterion_kd = L2Loss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _nstloss(s_features, t_features, weight, **kwargs):
            """Neuron selectivity transfer. https://arxiv.org/abs/1707.01219"""
            criterion_kd = NSTLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _rkdloss(s_features, t_features, weight, **kwargs):
            """Relational Knowledge Disitllation. https://arxiv.org/pdf/1904.05068.pdf"""
            criterion_kd = RKDLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _sploss(s_features, t_features, weight, **kwargs):
            """Similarity Preserving. https://arxiv.org/pdf/1907.09682.pdf"""
            criterion_kd = SPLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        def _crdloss(s_features, t_features, weight, **kwargs):
            """CRD loss. https://arxiv.org/abs/1910.10699"""
            criterion_kd = CRDLoss(**kwargs)
            return weight * criterion_kd(s_features, t_features)

        self.loss_function_map = {
            # "ab": _abloss,
            # "at": _atloss,
            # "cc": _ccloss,
            # "cd": _cdloss,
            # "ft": _ftloss,
            # "js": _jsloss,
            "kd": _kdloss,
            "ce": _celoss,
            "kl": _klloss,
            "l2": _l2loss,
            # "nst": _nstloss,
            # "rkd": _rkdloss,
            # "crd": _crdloss,
            # "sp": _sploss,
        }

        self.loss_cls_map = {
            "kd": KDLoss,
            "ce": CELoss,
            "kl": KLLoss,
            "l2": L2Loss
        }

    def clear(self):
        """End variables lifetime within mimic() function."""
        self.s_output_map = {}
        self.t_output_map = {}
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._remove_input()

    def _length_check(self, lists, check_type_name, names):
        """Check whether two lists lengths are equal and non-zero.

        Args:
            lists: [list_1, list_2]. Two lists to check the length.
            check_type_name: The type name of this check. For example, layer name length check.
            names: [name_1, name_2]. The list names corresponding to the lists.

        Raises:
            unmatched: two lists lengths are unequal.
            empty: lists are empty.
        """
        if len(lists[0]) != len(lists[1]):
            raise Exception("{} unmatched. {} = {} VS {} = {}".format(
                check_type_name, names[0], len(lists[0]), names[1], len(lists[1])))
        if not lists[0]:
            raise Exception("{} is empty.".format(check_type_name))

    def _type_check(self, mimic_config):
        """Check the whether mimic_type is defined and valid:

        Args:
            mimic_config: a list of mimic_job. Can be defined in YAML file.

        Raises:
            undefined: lacking "type" keyword in mimic_job config.
            unsupported: unsupported mimic_type of the class.
        """
        for i, mimic_job in enumerate(mimic_config):
            mimic_type = mimic_job.get('type', None)
            if mimic_type is None:
                message = 'mimic type undefined.' + \
                          'ID = {}, Class = {}'.format(i, self.__class__.__name__)
                raise Exception(message)
            if not self.is_supported_type(mimic_type):
                message = 'mimic type {} unsupported.'.format(mimic_type) + \
                          'ID = {}, Class = {}'.format(i, self.__class__.__name__)
                raise Exception(message)

    def _sanity_check(self, mimic_config):
        """Check whether the mimic config is correct.

        Args:
            mimic_config: a list of mimic_job. Can be defined in YAML file.
        """
        self._type_check(mimic_config)
        self._config_check(mimic_config)

    def _find_module(self, model, layer_name):
        """Find module(layer) by name.

        Args:
            model: the whole pytorch model.
            layer_name: the layer's full name in dotted string form.

        Returns:
            module: the module whose full name equals layer_name

        Raises:
            doesn't exist: cannot find the layer with the given name. Please check layer_name again.
        """
        if not layer_name:
            return model

        split_name = layer_name.split('.')
        module = model
        is_found = True
        for i, part_name in enumerate(split_name):
            is_found = False
            for child_name, child_module in module.named_children():
                if part_name == child_name:
                    module = child_module
                    is_found = True
                    break
            if not is_found:
                raise Exception("layer_name {} doesn't exist".format(layer_name))
        return module

    def _register_hooks(self, model, layer_name, output_map):
        """Register output hooks in model.

        This function register a hook in the specific module of the pytorch model. In this way, output_map is
        constructed which is used do translation between name and features to input. No additional GPU memory
        is used during this process.

        Args:
            model: the pytorch model.
            layer_name: to point the specific module.
            output_map: the specific output_map to register.
        """
        def hook(module, input, output):
            output_map[layer_name] = output

        module = self._find_module(model, layer_name)

        if layer_name not in output_map:
            self.handles.append(module.register_forward_hook(hook))

    def _forward_model(self):
        """Feed the input and run the model.

        This function get the models' outputs, if applicable. And it add the output to output_map for mimic useage.
        $$$$s_output, and $$$$t_output represents students' output and teachers' respectively. Watch out. No layer
        name can be either of these two.

        Returns:
            s_output: student model output. Can be None if student model is unavailable in the class.
            t_output: teacher model output. Can be None if teacher model is unavailable in the class.
        """
        s_output, t_output = self._get_model_output()
        self.s_output_map['$$$$s_output'] = s_output
        self.t_output_map['$$$$t_output'] = t_output
        return s_output, t_output

    def _knowledge_distill_loss(self, mimic_id, mimic_job, **kwargs):
        """Get mimic loss of a mimic_job.

        Args:
            mimic_id: the id of mimic_job in mimic_config.
            mimic_job: the mimic_job defined by user.
        """
        s_features, t_features = self._build_distill_features(mimic_id, mimic_job)
        if self.adjust_layer is None:
            self.adjust_layer = AdjustFeatures(s_features, t_features, mimic_job.get('adjust_config', None))
            task_optimizer = self.task_helper.get_optimizer()
            optim_params = task_optimizer.state_dict()['param_groups']
            new_params = copy.deepcopy(optim_params[0])
            new_params['params'] = self.adjust_layer.parameters()
            self.task_helper.get_optimizer().add_param_group(new_params)
        s_features, t_features = self.adjust_layer(s_features, t_features)
        # Remove extra kwargs.
        mimic_kwargs = mimic_job['kwargs'].copy()
        mimic_weight = mimic_job['kwargs']['weight']
        if 'student_layer_name' in mimic_kwargs:
            del mimic_kwargs['student_layer_name']
        if 'teacher_layer_name' in mimic_kwargs:
            del mimic_kwargs['teacher_layer_name']
        if 'weight' in mimic_kwargs:
            del mimic_kwargs['weight']

        custom_info = 'mimic type: {}'.format(mimic_job['type'])
        # threading.Thread(target=send, args=(custom_info,))

        mimic_job_func = self.mimic_job_loss.get(mimic_job['name'], None)
        if mimic_job_func is None:
            self.mimic_job_loss[mimic_job['name']] = self.loss_cls_map[mimic_job['type']](**mimic_kwargs)

        return mimic_weight * self.mimic_job_loss[mimic_job['name']](s_features, t_features, **kwargs)

        # return self.loss_function_map[mimic_job['type']](
        #     s_features, t_features, **mimic_kwargs)

    def is_supported_type(self, mimic_type):
        """Base supported type. Can be override by child class to do constrans or extending."""
        return mimic_type in self.loss_function_map


class ModelMimicModel(BaseDistiller):
    """Class for student model mimicing teacher model.

    You should provide both student model and teacher model. During the training process, two models
    are forwarded one by one. Although no extra GPU memory cost is not introduced, teacher model can be
    somehow too large. So, be very careful of the GPU memory.
    """
    def __init__(self, model, model_teacher, task_helper=None):
        """Init ModelMimicModel.
        Args:
            model: student model to train.
            model_teacher: teacher model as the mimic target.
        """
        super().__init__(task_helper=task_helper)
        self.model = model
        self.model_teacher = model_teacher

    def _register_input(self, input_var):
        """Register necessary variables.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        self.input_var = input_var

    def _remove_input(self):
        """Remove registered variables.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        self.input_var = None

    def _config_check(self, mimic_config):
        """Check mimic_config.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        for i, mimic_job in enumerate(mimic_config):
            # Get type and kwargs
            mimic_type = mimic_job['type']
            mimic_kwargs = mimic_job.get('kwargs', None)
            if mimic_kwargs is None:
                raise KeyError('Cannot find "kwargs" in config')

            # Check kwargs
            if 'weight' not in mimic_kwargs:
                raise KeyError('Cannot find "weight" at <job {}, type {}>'.format(i, mimic_type))
            if mimic_type == 'kd':
                if 'temperature' not in mimic_kwargs:
                    raise KeyError('Cannot find "temperature" at <job {}, type {}>'.format(i, mimic_type))

            # Check feature numbers
            s_layer_names = mimic_kwargs.get('student_layer_name', [])
            t_layer_names = mimic_kwargs.get('teacher_layer_name', [])
            check_type_name = 'feature number at <job {}, type {}>'.format(i, mimic_type)
            if mimic_type == 'kd':
                if not s_layer_names:
                    s_layer_names = ['$$$$s_output']
                if not t_layer_names:
                    t_layer_names = ['$$$$t_output']
            self._length_check(
                [s_layer_names, t_layer_names],
                check_type_name, ['teacher features', 'student layer names'])

    def _build_output_map(self, mimic_config):
        """Build output_map of student and teacher.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        for mimic_job in mimic_config:
            for s_layer_name in mimic_job['kwargs'].get('student_layer_name', []):
                self._register_hooks(self.model, s_layer_name, self.s_output_map)

            for t_layer_name in mimic_job['kwargs'].get('teacher_layer_name', []):
                self._register_hooks(self.model_teacher, t_layer_name, self.t_output_map)

    def _get_model_output(self):
        """Forward model and get output.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        s_output = self.model(self.input_var)
        with torch.no_grad():
            t_output = self.model_teacher(self.input_var)
        return s_output, t_output

    def _build_distill_features(self, mimic_id, mimic_job):
        """Prepare features as the input of distill loss functions.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        s_features = []
        t_features = []

        for s_layer_name in mimic_job['kwargs'].get('student_layer_name', []):
            s_features.append(self.s_output_map[s_layer_name])
        for t_layer_name in mimic_job['kwargs'].get('teacher_layer_name', []):
            t_features.append(self.t_output_map[t_layer_name])

        if mimic_job['type'] == 'kd':
            if not s_features:
                s_features.append(self.s_output_map['$$$$s_output'])
            if not t_features:
                t_features.append(self.t_output_map['$$$$t_output'])
        return s_features, t_features

    def mimic(self, input_var, mimic_config):
        """Main function for mimicing.

        Args:
            input_var: inputs for student model and teacher model.
            mimic_config: one mimic_job or a list of mimic_jobs.

        Returns:
            loss: loss of mimicing.
            losses: list of loss of each mimicing job.
            s_output: output of student model.
            t_output: output of teacher model.
        """
        if not isinstance(mimic_config, list):
            mimic_config = [mimic_config]

        self._register_input(input_var=input_var)

        self._sanity_check(mimic_config)

        self._build_output_map(mimic_config)

        s_output, t_output = self._forward_model()

        losses = []
        for mimic_id, mimic_job in enumerate(mimic_config):
            loss = self._knowledge_distill_loss(mimic_id, mimic_job)
            losses.append(loss)

        self.clear()
        return sum(losses), losses, s_output, t_output


class ModelMimicFeature(BaseDistiller):
    """Class for student model mimicing given features.

    You should provide student model and features to mimic. The feature list should be exactily the
    same order as mimic_job. In each list, features' order should match layer names' order.
    For example:
        layer_names = [[l1], [l2, l3, l4]]
        features = [[f1], [f2, f3, f4]]
        mimic_config = [mimic_job1, mimic_job2]
    where
        [l1]--[f1]: mimic_job1 define [l1]
        [l2, l3, l3]--[f2, f3, f4]: mimic_job2 define [l2, l3, l3]

    You can also provides one mimic_job only:
        layer_names = [l1, l2, l3]
        features = [f1, f2, f3]
        mimic_config = mimic_job1
    And the matched pairs are:
        l1--f1, l2--f2, l3--f3
    """
    def __init__(self, model, task_helper=None):
        """Init ModelMimicFeature.
        Args:
            model: student model to train.
        """
        super().__init__(task_helper=task_helper)
        self.model = model

    def _register_input(self, input_var, t_feature_groups):
        """Register necessary variables.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        self.input_var = input_var
        self.t_feature_groups = t_feature_groups

    def _remove_input(self):
        """Remove registered variables.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        self.input_var = None
        self.t_feature_groups = None

    def _config_check(self, mimic_config):
        """Check mimic_config.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        check_type_name = 'feature group number'
        self._length_check(
            [self.t_feature_groups, mimic_config],
            check_type_name, ['teacher features', 'mimic jobs'])

        for i, mimic_job in enumerate(mimic_config):
            # Get type and kwargs
            mimic_type = mimic_job['type']
            mimic_kwargs = mimic_job.get('kwargs', None)
            if mimic_kwargs is None:
                raise KeyError('Cannot find "kwargs" in config')

            # Check kwargs
            if 'weight' not in mimic_kwargs:
                message = 'Cannot find "weight"' + 'at <job {}, type {}>'.format(i, mimic_type)
                raise KeyError(message)
            if mimic_type == 'kd':
                if 'temperature' not in mimic_kwargs:
                    message = 'Cannot find "temperature"' + 'at <job {}, type {}>'.format(i, mimic_type)
                    raise KeyError(message)
            if 'teacher_layer_name' in mimic_kwargs:
                message = "Unsupported keyword teacher_layer_name in {}".format(self.__class__.__name__) + \
                          'at <job {}, type {}>'.format(i, mimic_type)
                raise KeyError(message)

            # Check feature numbers
            s_layer_names = mimic_kwargs.get('student_layer_name', [])
            check_type_name = 'feature number at <job {}, type {}>'.format(i, mimic_type)
            if mimic_type == 'kd':
                if not s_layer_names:
                    s_layer_names = ['$$$$s_output']
            self._length_check(
                [self.t_feature_groups[i], s_layer_names],
                check_type_name, ['teacher features', 'student layer names'])

    def _build_output_map(self, mimic_config):
        """Build output_map of student.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        for mimic_job in mimic_config:
            for s_layer_names in mimic_job['kwargs'].get('student_layer_name', []):
                self._register_hooks(self.model, s_layer_names, self.s_output_map)

    def _get_model_output(self):
        """Forward model and get output.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        s_output = self.model(self.input_var)
        t_output = None
        return s_output, t_output

    def _build_distill_features(self, mimic_id, mimic_job):
        """Prepare features as the input of distill loss functions.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        s_features = []
        t_features = self.t_feature_groups[mimic_id]

        for s_layer_name in mimic_job['kwargs'].get('student_layer_name', []):
            s_features.append(self.s_output_map[s_layer_name])

        if mimic_job['type'] == 'kd':
            if not s_features:
                s_features.append(self.s_output_map['$$$$s_output'])

        return s_features, t_features

    def mimic(self, input_var, t_feature_groups, mimic_config):
        """Main function for mimicing.

        Args:
            input_var: inputs for student model and teacher model.
            t_feature_groups: groups of feature lists as teacher. e.g. [[f1], [f2, f3]]
            mimic_config: one mimic_job or a list of mimic_jobs.

        Returns:
            loss: loss of mimicing.
            losses: list of loss of each mimicing job.
            s_output: output of student model.
            t_output: None.
        """
        if not isinstance(mimic_config, list):
            mimic_config = [mimic_config]
            t_feature_groups = [t_feature_groups]

        self._register_input(input_var=input_var, t_feature_groups=t_feature_groups)

        self._sanity_check(mimic_config)

        self._build_output_map(mimic_config)

        s_output, t_output = self._forward_model()

        losses = []
        for mimic_id, mimic_job in enumerate(mimic_config):
            loss = self._knowledge_distill_loss(mimic_id, mimic_job)
            losses.append(loss)

        self.clear()
        return sum(losses), losses, s_output, t_output


class FeatureMimicFeature(BaseDistiller):
    """Class for getting feature mimicing result.

    This class does not support KD type. You should provide both student
    and teacher features, as well as the list of mimic types.
    For example:
        student_features: [[s1], [s2, s3]]
        teacher_features: [[t1], [t2, t3]]
        mimic_types: [type1, type2]
    where
        [s1]--[t1]: perform type1 mimicing
        [s2, s3]--[t2, t3]: perform type2 mimicing
    Or you can do one type mimicing only:
        student_features: [s1, s2, s3]
        teacher_features: [t1, t2, t3]
        mimic_types: type1
    Features are matched as s1--t1, s2--t2, s3--t3
    """
    def __init__(self, task_helper=None):
        """Init FeatureMimicFeature."""
        super().__init__(task_helper=task_helper)

    def _register_input(self, s_feature_groups, t_feature_groups):
        """Register necessary variables.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        self.s_feature_groups = s_feature_groups
        self.t_feature_groups = t_feature_groups

    def _remove_input(self):
        """Remove registered variables.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        self.s_feature_groups = None
        self.t_feature_groups = None

    def _config_check(self, mimic_config):
        """Check mimic_config.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        # Check number of mimic groups
        check_type_name = 'feature group number'
        self._length_check(
            [self.s_feature_groups, self.t_feature_groups],
            check_type_name, ['student', 'teacher'])
        check_type_name = 'mimic type number'
        self._length_check(
            [self.s_feature_groups, mimic_config],
            check_type_name, ['features', 'types'])

        # Check number of features within a group
        for i, mimic_job in enumerate(mimic_config):
            mimic_type = mimic_job['type']
            check_type_name = 'feature number at <job {}, type {}>'.format(i, mimic_type)
            self._length_check(
                [self.s_feature_groups[i], self.t_feature_groups[i]],
                check_type_name, ['student', 'teacher'])

    def _build_output_map(self, mimic_config):
        """No need to build output_map.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        pass  # no output map needs to build

    def _get_model_output(self):
        """No output needs to get.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        s_output = t_output = None
        return s_output, t_output

    def _build_distill_features(self, mimic_id, mimic_job):
        """Prepare features as the input of distill loss functions.

        YOU MAY NOT CALL THIS FUNCTION.
        """
        s_features = self.s_feature_groups[mimic_id]
        t_features = self.t_feature_groups[mimic_id]
        return s_features, t_features

    def mimic(self, s_feature_groups, t_feature_groups, mimic_jobs, **kwargs):
        """Main function for mimicing.

        Args:
            s_feature_groups: groups of feature lists as teacher. e.g. [[s1], [s2, s3]]
            t_feature_groups: groups of feature lists as teacher. e.g. [[t1], [t2, t3]]
            mimic_jobs: one mimic_job or a list of mimic_jobs.

        Returns:
            loss: loss of mimicing.
            losses: list of loss of each mimicing job. You may want to do weighted sum of
                    mimicing losses on your own.
            s_output: None.
            t_output: None.
        """
        if not isinstance(mimic_jobs, list):
            mimic_jobs = [mimic_jobs]
            s_feature_groups = [s_feature_groups]
            t_feature_groups = [t_feature_groups]

        # Build mimic_config from mimic_jobs
        mimic_config = []
        for mimic_job in mimic_jobs:
            mimic_config.append(mimic_job)

        self._register_input(s_feature_groups=s_feature_groups, t_feature_groups=t_feature_groups)

        self._sanity_check(mimic_config)

        self._build_output_map(mimic_config)

        s_output, t_output = self._forward_model()

        losses = []
        for mimic_id, mimic_job in enumerate(mimic_config):
            loss = self._knowledge_distill_loss(mimic_id, mimic_job, **kwargs)
            losses.append(loss)

        self.clear()
        return sum(losses), losses, s_output, t_output
