def print_progress(recent, max):
    percent = '%.2f%%' % ((recent+1) / max * 100)
    percent_str = int(float(percent[:-1]))*"#" + (100 - int(float(percent[:-1]))) * "-" + percent
    print(f"\r{percent_str}", end="")
    if float(percent[:-1]) >= 100.0:
        print('\n')



def train_print_progress(epoch, max, loss):
    percent = '%.2f%%' % ((epoch+1) / max * 100)
    percent_str = int(float(percent[:-1]))*"#" + (100 - int(float(percent[:-1]))) * "-" + f" minibatch: {epoch} Loss: {loss}" 
    print(f"\r{percent_str}", end="")
    if float(percent[:-1]) >= 100.0:
        print('\n')
