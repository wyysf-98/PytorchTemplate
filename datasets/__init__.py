import imp

def get_dataset(cfg, split):
    dataset_module = 'datasets.' + cfg.datasets.type
    dataset_path = 'datasets/' + cfg.datasets.type + '.py'
    return imp.load_source(dataset_module, dataset_path).Dataset(cfg, split)
    # except Exception as e:
    #     print(e)
