import run
import thumb_32x32
import sys

def main(arg):
    save_path = run.main(arg)
    thumb_32x32.main(save_path)
    pass

if __name__ == '__main__':
    main(sys.argv[1:])