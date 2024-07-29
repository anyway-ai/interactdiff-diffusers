from .catalog import DatasetCatalog
import torch
import importlib

def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

class ConCatDataset():
    def __init__(self, dataset_name_list, ROOT, train=True, repeats=None):
        self.datasets = []
        cul_previous_dataset_length = 0
        offset_map = []
        which_dataset = []

        if repeats is None:
            repeats = [1] * len(dataset_name_list)
        else:
            assert len(repeats) == len(dataset_name_list)

        Catalog = DatasetCatalog(ROOT)
        for dataset_idx, (dataset_name, yaml_params) in enumerate(dataset_name_list.items()):
            repeat = repeats[dataset_idx]

            dataset_dict = getattr(Catalog, dataset_name)

            target = dataset_dict['target']
            params = dataset_dict['train_params'] if train else dataset_dict['val_params']
            if yaml_params is not None:
                params.update(yaml_params)
            dataset = instantiate_from_config(dict(target=target, params=params))

            self.datasets.append(dataset)
            for _ in range(repeat):
                offset_map.append(torch.ones(len(dataset)) * cul_previous_dataset_length)
                which_dataset.append(torch.ones(len(dataset)) * dataset_idx)
                cul_previous_dataset_length += len(dataset)
        offset_map = torch.cat(offset_map, dim=0).long()
        self.total_length = cul_previous_dataset_length

        self.mapping = torch.arange(self.total_length) - offset_map
        self.which_dataset = torch.cat(which_dataset, dim=0).long()

    def total_images(self):
        count = 0
        for dataset in self.datasets:
            print(dataset.total_images())
            count += dataset.total_images()
        return count

    def __getitem__(self, idx):
        dataset = self.datasets[self.which_dataset[idx]]
        return dataset[self.mapping[idx]]

    def __len__(self):
        return self.total_length
