import os
import re
import json
import yaml
from pathlib import Path
from datetime import datetime
from .tools import set_seed, dict_merge

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

class ConfigParser():
    def __init__(self, args):
        """
        class to parse configuration .yaml file. Create job folder if in 'train' mode.
        """
        args = args.parse_args()
        self.config = self._convert_to_dotdict(self._parse_config(args.config))
        self.name = args.name if args.name is not None else datetime.now().strftime('%m%d_%H%M%S')
        self.mode = args.mode
        self.config['seed'] = args.seed

        output_dir = Path(self.config['output_dir'])
        exp_name = self.config['exp_name']
        self.save_dir = output_dir / exp_name / self.name

        if args.seed is not None:
            set_seed(args.seed)

    def dump(self, save_path):
        """
        save d with Dict format to $save_path.
        """
        with open(save_path, 'w') as f:
            f.write(yaml.dump(self.config))
        f.close()

    def _parse_config(self, config):
        all_base_cfg = dotdict()
        with open(config, 'r') as f:
            cfg = yaml.safe_load(f)
        f.close()

        # NOTE: cfgs outside have higher priority than cfgs in _BASE_
        BASE_KEY = '_BASE_'
        if BASE_KEY in cfg:
            base_ymls = list(cfg[BASE_KEY])
            for base_yml in base_ymls:
                if base_yml.startswith("~"):
                    base_yml = os.path.expanduser(base_yml)
                if not base_yml.startswith('/'):
                    base_yml = os.path.join(os.path.dirname(config), base_yml)
                base_cfg = self._parse_config(base_yml)
                all_base_cfg = dict_merge(all_base_cfg, base_cfg)

            del cfg[BASE_KEY]
            return dict_merge(all_base_cfg, cfg)

        return cfg

    def _convert_to_dotdict(self, config):
        for k, v in config.items():
            if isinstance(v, dict):
                config[k] = self._convert_to_dotdict(v)
        return dotdict(config)
    
    def __getattr__(self, name):
        """
        
        Access items use dot.notation.
        """
        return self.config[name]

    def __getitem__(self, name):
        """
        
        Access items like ordinary dict.
        """
        return self.config[name]

    def __str__(self):
        """
        print config dict with pretty format.
        """
        head_str = '\n' + '*'*20+'  Config  '+'*'*20
        end_str = '='*50

        # remove punctuation for better print.
        _remove_punctuation = lambda text: re.sub(r'[{}]+'.format('{!,;?"\'、，；}'), ' ', text)
        config_str = head_str + '\n' + _remove_punctuation(json.dumps(self.config, indent=2, ensure_ascii=False)) + '\n' + end_str + '\n'
        return config_str
