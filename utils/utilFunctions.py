def writeLogs(str):
    f = open('./train_logs.txt', 'a')
    f.write(str + '\n')
    f.close()

def writeLogs_KCV(str):
    f = open('./train_logs_kcv.txt', 'a')
    f.write(str + '\n')
    f.close()
