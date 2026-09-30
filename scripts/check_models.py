#!/usr/bin/env python3
"""Small paid multimodal gateway smoke test. Never runs on import."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dotenv import load_dotenv
from models import call_model
from models.catalog import NVIDIA_MODELS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', action='append', help='Repeat for multiple models')
    parser.add_argument('--all', action='store_true')
    args = parser.parse_args()
    load_dotenv(Path(__file__).resolve().parents[1] / '.env')
    names = list(NVIDIA_MODELS) if args.all else args.model
    if not names:
        parser.error('Specify --model or --all; these calls may incur charges.')
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 255
    failed = False
    for name in names:
        try:
            text = call_model(
                images=[image], prompt='What color fills the image? Reply with the color only.',
                system_prompt='Answer visual questions concisely.', model_name=name,
                n_tokens=1024, extra_messages=[
                    {'role': 'user', 'text': 'I will show you an image.'},
                    {'role': 'assistant', 'text': 'Ready.'}])
            ok = 'red' in text.lower()
            failed |= not ok
            print(json.dumps({'model': name, 'ok': ok, 'response': text[:200]}), flush=True)
        except Exception as exc:
            failed = True
            print(json.dumps({'model': name, 'ok': False, 'error': str(exc)[:500]}), flush=True)
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
