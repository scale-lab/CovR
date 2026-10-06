import re 
import os 
import sys
import argparse
import random
import logging
import numpy as np

from colorama import Fore, Style, init
from pyverilog.vparser.parser import parse


init(autoreset=True)

class ColorFormatter(logging.Formatter):
    FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    FORMATS = {
        logging.DEBUG: Fore.CYAN + FORMAT + Style.RESET_ALL,
        logging.INFO: Fore.GREEN + FORMAT + Style.RESET_ALL,
        logging.WARNING: Fore.YELLOW + FORMAT + Style.RESET_ALL,
        logging.ERROR: Fore.RED + FORMAT + Style.RESET_ALL,
        logging.CRITICAL: Fore.MAGENTA + FORMAT + Style.RESET_ALL
    }

    def format(self, record):
        if record.levelno == logging.INFO and ':' in record.msg:
            # Extract color from the message
            color, message = record.msg.split(':', 1)
            color_attr = getattr(Fore, color.upper(), Fore.WHITE)
            record.msg = message.strip()
            log_fmt = color_attr + self.FORMAT + Style.RESET_ALL
        else:
            log_fmt = self.FORMATS.get(record.levelno, self.FORMAT)
        
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)



def get_logger(output):
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=output, level=logging.INFO, filemode='w')
    logging.getLogger("httpx").setLevel(logging.WARNING)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(ColorFormatter())

    file_handler = logging.FileHandler(output)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(ColorFormatter())

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def write_answers(tmp_dir, problem: str, metric:str, answers: list[str]) -> list[str]:
    answer_list = []
    for j, sample in enumerate(answers): 
        file_path = os.path.join(tmp_dir, f"{problem}_ans_{metric}__{j}.v")
        answer_list.append(file_path)
        with open(file_path, "w") as file:
            if sample is None:
                sample = ""
            file.write(sample)

    return answer_list


def write_text_answers(tmp_dir, problems: list[str], answers: list[str], metric: str, num_samples: int) -> list[str]:
    total_samples = len(answers)
    for b in range(0, total_samples, num_samples):
        for i in range(b, min(b + num_samples, total_samples)):
            sample = answers[i]
            file_path = os.path.join(tmp_dir, f"{problems[b % num_samples]}_ans_txt_{metric}__{i}.txt")
            with open(file_path, "w") as file:
                file.write(sample)


def write_ground_truth(tmp_dir, problem: str, ground_truth: list[str]) -> list[str]:
    ground_truth_files = []
    for i, verilog in enumerate(ground_truth): 
        file_path = os.path.join(tmp_dir, f"{problem}_ground_truth_{i}.v")
        ground_truth_files.append(file_path)
        with open(file_path, "w") as file:
            file.write(verilog)
    
    return ground_truth_files


def write_ground_truth_files(tmp_dir, problems: list[str], ground_truth: list[str]) -> list[str]:
    ground_truth_files = []
    for i, (problem, verilog) in enumerate(zip(problems, ground_truth)): 
        file_path = os.path.join(tmp_dir, f"{problem}_ground_truth_{i}.v")
        ground_truth_files.append(file_path)
        with open(file_path, "w") as file:
            file.write(verilog)
    
    return ground_truth_files