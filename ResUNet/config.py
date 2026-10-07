"""Record the effective CLI/config settings beside each output."""
import json
from pathlib import Path


def save_config(opt, directory):
    settings = vars(opt).copy()
    # The source config is not needed to reuse the resolved settings.
    settings.pop('config', None)
    if 'mask_seed' in settings and settings['mask_seed'] is None:
        settings['mask_seed'] = settings['seed']
    path = Path(directory) / 'config.json'
    path.write_text(json.dumps(settings, indent=2, ensure_ascii=False, allow_nan=False) + '\n',
                    encoding='utf-8')
    return path
