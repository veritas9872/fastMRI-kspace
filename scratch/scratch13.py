from time import sleep

if __name__ == '__main__':
    try:
        sleep(10)
    except KeyboardInterrupt:
        print('It worked!')
    else:
        print('Finished!')
