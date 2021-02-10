import subprocess
import os
import signal
import time
# from threading  import Thread
import threading
import multiprocessing
from pyabc.visserver import server_dash
# def test_server_dash():
# cmd = "abc-server-dash"
# proc = subprocess.Popen([cmd], stdout=subprocess.PIPE,
#                        shell=True, preexec_fn=os.setsid)
# time.sleep(2)
# os.killpg(os.getpgid(proc.pid),
#            signal.SIGTERM)  # Send the signal to all the process groups
# print("proc: ",proc)

# flag = False

def kill_dash_server():
    time.sleep(5)
try:
    t1 = multiprocessing.Process(target=server_dash.run_app)
    t1.start()
    time.sleep(3)

except:
    print('Dash server did not start properly')
t1.terminate()

# t2.start()
# os.killpg(os.getpgid(proc.pid),
#            signal.SIGTERM)  # Send the signal to all the process groups
# print("proc: ",proc)



#
# import time
# import threading
# def calc_square(number, threadName):
#    print("Calculate square numbers: ")
#    for n in number:
#       time.sleep(0.2)   #artificial time-delay
#       print("Hi, I'm thread: ", threadName,'. square: ', str(n*n))
# def calc_cube(number, threadName):
#    print("Calculate cude numbers: ")
#    for n in number:
#       time.sleep(0.2)
#       print("Hi, I'm thread: ", threadName,'. cube: ', str(n*n*n))
# arr = [2,3,8,9]
# t = time.time()
# t1 = threading.Thread(target = calc_square,args=(arr,1))
# t2 = threading.Thread(target = calc_cube,args=(arr,2))
# # creating two threads here t1 & t2
# t1.start()
# t2.start()
# # starting threads here parallelly by usign start function.
# t1.join()
# # this join() will wait until the cal_square() function is finised.
# t2.join()
# # this join() will wait unit the cal_cube() function is finised.
# print("Successed!")
