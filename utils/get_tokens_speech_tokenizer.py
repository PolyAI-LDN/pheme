"""Get tokens using the SpeechTokenizer.

Apply SpeechTokenizer to extract acoustic and semantic tokens. 
The tokens will be extracted to 
encoding_output/acoustic and encoding_output/semantic.

python utils/get_tokens_speech_tokenizer.py \
    --config_path ckpt/speechtokenizer/config.json \
    --ckpt_path ckpt/speechtokenizer/SpeechTokenizer.pt \
    --encoding_input datasets/example/audios \
    --encoding_output datasets/example/audios-speech-tokenizer

Copyright PolyAI Limited.
"""
import argparse
import pathlib

from modules.speech_tokenizer import SpeechTokenizer

PROJECT_ROOT = str(pathlib.Path(__file__).parent.parent.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to the SpeechTokenizer config",
        default=PROJECT_ROOT + "/ckpt/speechtokenizer/config.json",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        help="Path to the SpeechTokenizer checkpoint",
        default=PROJECT_ROOT + "/ckpt/speechtokenizer/SpeechTokenizer.pt",
    )
    parser.add_argument(
        "--encoding_input",
        type=str,
        help="Path to the input folder for encoding",
        default=PROJECT_ROOT + "/datasets/example/audios",
    )
    parser.add_argument(
        "--encoding_output",
        type=str,
        help="Path where to save the encoded tokens",
        default=PROJECT_ROOT + "/datasets/example/audios-speech-tokenizer"
    )
    parser.add_argument(
        "--start_percent",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--end_percent",
        type=int,
        default=100,
    )

    args = parser.parse_args()
    print("Parsed args")
    print(args)

    tokenizer = SpeechTokenizer(
        config_path=args.config_path,
        ckpt_path=args.ckpt_path,
    )
    #tokenizer.encode_files_with_model_seq
    # TODO: debug execution speed, utilize multi-gpus
    tokenizer.encode_files_with_model_concurrent(
        folder_path=args.encoding_input, destination_folder=args.encoding_output,
        start_percent=args.start_percent, end_percent=args.end_percent
    )
