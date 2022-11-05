import logging 
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('batch_run')
from main import create_parser, main
import glob

def create_batch_runner_parser():
    import argparse
    batch_runner_arg_parser = argparse.ArgumentParser()
    batch_runner_arg_parser.add_argument('audio_file_regex', type=str, metavar='audio_file_regex',
                    help='The file path of dir containing audio to be transcribed.')
    batch_runner_arg_parser.add_argument('-a', '--asc_model_fp', type=str, metavar='asc_model_fp', default='models/cnn_normmc/ascending.npz',
                    help='The name of the ascending model.')
    batch_runner_arg_parser.add_argument('-d', '--desc_model_fp', type=str, metavar='desc_model_fp', default='models/cnn_normmc/descending.npz',
                    help='The name of the descending model.')
    batch_runner_arg_parser.add_argument('-o', '--output_dir', type=str, metavar='output_dir', default='outputs',
                    help='The output directory.')
    batch_runner_arg_parser.add_argument('-m', '--melody_contour', type=str, default=None, 
                    help='The filepath of melody contour.')
    batch_runner_arg_parser.add_argument('-e', '--evaluate', type=str, default=None, 
                    help='The filepath of answer file.')
                    
    return batch_runner_arg_parser

if __name__ == "__main__":
    batch_runner_arg_parser = create_batch_runner_parser()
    batch_args = batch_runner_arg_parser.parse_args()

    logger.info(f'Found matched audio file in {batch_args.audio_file_regex}')
    audio_file_list = []
    for file_path in glob.glob(batch_args.audio_file_regex):
        logger.info(file_path)
        audio_file_list.append(file_path)
    
    melody_contour = None if batch_args.melody_contour == 'None' else batch_args.melody_contour
    evaluate = None if batch_args.evaluate == 'None' else batch_args.evaluate
    for audio_path in audio_file_list:
        parser = create_parser()
        args = parser.parse_args([
                '--asc_model_fp', 
                batch_args.asc_model_fp,
                '--desc_model_fp',
                batch_args.desc_model_fp,
                '--output_dir', 
                batch_args.output_dir,
                '--melody_contour',
                melody_contour,
                '--evaluate',
                evaluate,
                audio_path
            ])
        logger.info(f'args for each audio file: {audio_path}')
        logger.info(args)

        logger.info('start to transcribe....')
        main(args.audio_fp, args.asc_model_fp, args.desc_model_fp, 
             args.output_dir, melody_contour, evaluate)
