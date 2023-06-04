'''
Main file for RG images anonymization 

Author: Luiz Coelho
Data: April 2023
'''

from anonymizer import Anonymizer
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', default='./config.json')
    args = parser.parse_args()
    
    if not os.path.isfile(args.config):
        print(f"\"{args.config}\" is not a readable file.")

    with open(args.config) as f:
        cfg = json.load(f)


    samples = cfg['samples']

    base_dir = cfg['input_dir']
    save_dir = cfg['save_dir']

    for sample in samples:
        sample_cfg = cfg['sample_cfg'][sample]

        anon = Anonymizer(
            base_dir + sample_cfg['images'],
            base_dir + sample_cfg['labels'],
            sample,
            cfg['gan_config'],
            filelist = base_dir + sample_cfg['filelist'],
            max_img_size = cfg['max_width']
        )
        anon.run()