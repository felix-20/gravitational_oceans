import os


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

PATH_TO_TEST_FOLDER = os.path.join(os.getcwd(), 'test_data')
PATH_TO_TRAIN_FOLDER = os.path.join(os.getcwd(), 'train_data')
PATH_TO_MODEL_FOLDER = os.path.join(os.getcwd(), 'models_saved')

# setup
if not os.path.isdir(PATH_TO_TRAIN_FOLDER):
    os.makedirs(PATH_TO_TRAIN_FOLDER)
if not os.path.isdir(PATH_TO_TEST_FOLDER):
    os.makedirs(PATH_TO_TEST_FOLDER)
if not os.path.isdir(PATH_TO_MODEL_FOLDER):
    os.makedirs(PATH_TO_MODEL_FOLDER)

def print_red(*text):
    print(f'{bcolors.FAIL}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


def print_blue(*text):
    print(f'{bcolors.OKCYAN}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


def print_green(*text):
    print(f'{bcolors.OKGREEN}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


def print_yellow(*text):
    print(f'{bcolors.WARNING}{" ".join([str(t) for t in text])}{bcolors.ENDC}')


if __name__ == '__main__':
    print_red('This', 'text', 'is red', 1, 23)
    print_blue('This', 'text', 'is blue', 1, 23)
    print_green('This', 'text', 'is green', 1, 23)
    print_yellow('This', 'text', 'is yellow', 1, 23)
