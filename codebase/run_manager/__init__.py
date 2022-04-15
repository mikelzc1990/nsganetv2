from codebase.data_providers.imagenet import *
from codebase.data_providers.cifar import *
from codebase.data_providers.flowers102 import *
from codebase.data_providers.stl10 import *
from codebase.data_providers.dtd import *
from codebase.data_providers.pets import *
from codebase.data_providers.aircraft import *

from ofa.imagenet_classification.run_manager.run_config import RunConfig


class ImagenetRunConfig(RunConfig):

    def __init__(self, n_epochs=1, init_lr=1e-4, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='imagenet', train_batch_size=128, test_batch_size=512, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224,
                 data_path='/mnt/datastore/ILSVRC2012',
                 **kwargs):
        super(ImagenetRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.imagenet_data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == ImagenetDataProvider.name():
                DataProviderClass = ImagenetDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.imagenet_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class CIFARRunConfig(RunConfig):
    def __init__(self, n_epochs=5, init_lr=0.01, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='cifar10', train_batch_size=96, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=2, resize_scale=0.08, distort_color=None, image_size=224,
                 data_path='/mnt/datastore/CIFAR',
                 **kwargs):
        super(CIFARRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.cifar_data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == CIFAR10DataProvider.name():
                DataProviderClass = CIFAR10DataProvider
            elif self.dataset == CIFAR100DataProvider.name():
                DataProviderClass = CIFAR100DataProvider
            elif self.dataset == CINIC10DataProvider.name():
                DataProviderClass = CINIC10DataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.cifar_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class Flowers102RunConfig(RunConfig):

    def __init__(self, n_epochs=3, init_lr=1e-2, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='flowers102', train_batch_size=32, test_batch_size=250, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=4, resize_scale=0.08, distort_color=None, image_size=224,
                 data_path='/mnt/datastore/Flowers102',
                 **kwargs):
        super(Flowers102RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.flowers102_data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == Flowers102DataProvider.name():
                DataProviderClass = Flowers102DataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.flowers102_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class STL10RunConfig(RunConfig):

    def __init__(self, n_epochs=5, init_lr=1e-2, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='stl10', train_batch_size=96, test_batch_size=256, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=4, resize_scale=0.08, distort_color=None, image_size=224,
                 data_path='/mnt/datastore/STL10',
                 **kwargs):
        super(STL10RunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )

        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.stl10_data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == STL10DataProvider.name():
                DataProviderClass = STL10DataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.stl10_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class DTDRunConfig(RunConfig):

    def __init__(self, n_epochs=1, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='dtd', train_batch_size=32, test_batch_size=250, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None, model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224,
                 data_path='/mnt/datastore/dtd',
                 **kwargs):
        super(DTDRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == DTDDataProvider.name():
                DataProviderClass = DTDDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class PetsRunConfig(RunConfig):

    def __init__(self, n_epochs=1, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='pets', train_batch_size=32, test_batch_size=250, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224,
                 data_path='/mnt/datastore/Oxford-IIITPets',
                 **kwargs):
        super(PetsRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.imagenet_data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == OxfordIIITPetsDataProvider.name():
                DataProviderClass = OxfordIIITPetsDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.imagenet_data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


class AircraftRunConfig(RunConfig):

    def __init__(self, n_epochs=1, init_lr=0.05, lr_schedule_type='cosine', lr_schedule_param=None,
                 dataset='aircraft', train_batch_size=32, test_batch_size=250, valid_size=None,
                 opt_type='sgd', opt_param=None, weight_decay=4e-5, label_smoothing=0.0, no_decay_keys=None,
                 mixup_alpha=None,
                 model_init='he_fout', validation_frequency=1, print_frequency=10,
                 n_worker=32, resize_scale=0.08, distort_color='tf', image_size=224,
                 data_path='/mnt/datastore/Aircraft',
                 **kwargs):
        super(AircraftRunConfig, self).__init__(
            n_epochs, init_lr, lr_schedule_type, lr_schedule_param,
            dataset, train_batch_size, test_batch_size, valid_size,
            opt_type, opt_param, weight_decay, label_smoothing, no_decay_keys,
            mixup_alpha,
            model_init, validation_frequency, print_frequency
        )
        self.n_worker = n_worker
        self.resize_scale = resize_scale
        self.distort_color = distort_color
        self.image_size = image_size
        self.data_path = data_path

    @property
    def data_provider(self):
        if self.__dict__.get('_data_provider', None) is None:
            if self.dataset == FGVCAircraftDataProvider.name():
                DataProviderClass = FGVCAircraftDataProvider
            else:
                raise NotImplementedError
            self.__dict__['_data_provider'] = DataProviderClass(
                save_path=self.data_path,
                train_batch_size=self.train_batch_size, test_batch_size=self.test_batch_size,
                valid_size=self.valid_size, n_worker=self.n_worker, resize_scale=self.resize_scale,
                distort_color=self.distort_color, image_size=self.image_size,
            )
        return self.__dict__['_data_provider']


def get_run_config(**kwargs):
    if kwargs['dataset'] == 'imagenet':
        run_config = ImagenetRunConfig(**kwargs)
    elif kwargs['dataset'].startswith('cifar') or kwargs['dataset'].startswith('cinic'):
        run_config = CIFARRunConfig(**kwargs)
    elif kwargs['dataset'] == 'flowers102':
        run_config = Flowers102RunConfig(**kwargs)
    elif kwargs['dataset'] == 'stl10':
        run_config = STL10RunConfig(**kwargs)
    elif kwargs['dataset'] == 'dtd':
        run_config = DTDRunConfig(**kwargs)
    elif kwargs['dataset'] == 'pets':
        run_config = PetsRunConfig(**kwargs)
    elif kwargs['dataset'] == 'aircraft':
        run_config = AircraftRunConfig(**kwargs)
    else:
        raise NotImplementedError

    return run_config


