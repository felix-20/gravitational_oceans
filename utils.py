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


def print_red(text: str):
    print(f'{bcolors.FAIL}{text}{bcolors.ENDC}')


def print_blue(text: str):
    print(f'{bcolors.OKCYAN}{text}{bcolors.ENDC}')


def print_green(text: str):
    print(f'{bcolors.OKGREEN}{text}{bcolors.ENDC}')

def print_yellow(text: str):
    print(f'{bcolors.WARNING}{text}{bcolors.ENDC}')
