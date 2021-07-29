import serial
import time
ser = serial.Serial("/dev/ttyUSB0", 9600, timeout=1)

def write_data(string):
    ser.write(string.encode())
    ser.close()
