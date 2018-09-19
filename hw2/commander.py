import subprocess

if __name__ == '__main__':
    f = open("commands.txt")
    for line in f.readlines():
        subprocess.call(line, shell=True)
    f.close()
