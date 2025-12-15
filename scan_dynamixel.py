#!/usr/bin/env python3
"""
Dynamixel Scanner Tool
Scans for Dynamixel motors across standard baudrates and IDs.
"""
import sys
import glob
from dynamixel_sdk import *

# Common Baudrates
BAUDRATES = [57600, 115200, 1000000, 2000000, 3000000, 4000000, 9600]
PROTOCOL_VERSION = 2.0

def get_available_ports():
    if sys.platform.startswith('linux'):
        return glob.glob('/dev/ttyUSB*') + glob.glob('/dev/ttyACM*')
    return []

def scan_port(port):
    print(f"\nScanning Port: {port}...")
    
    for baud in BAUDRATES:
        port_handler = PortHandler(port)
        packet_handler = PacketHandler(PROTOCOL_VERSION)
        
        if port_handler.openPort() and port_handler.setBaudRate(baud):
            print(f"  Baudrate: {baud} - Port Open. Pinging IDs 0-20...")
            found_motors = []
            
            for dxl_id in range(20): # Scan IDs 0 to 19 (Adjust if higher IDs needed)
                model_number, dxl_comm_result, dxl_error = packet_handler.ping(port_handler, dxl_id)
                if dxl_comm_result == COMM_SUCCESS:
                    print(f"    [FOUND] ID: {dxl_id}, Model: {model_number}")
                    found_motors.append(dxl_id)
            
            if found_motors:
                print(f"  >>> SUCCESS at {baud} bps! Found IDs: {found_motors}")
                port_handler.closePort()
                return baud, found_motors
                
            port_handler.closePort()
        else:
            print(f"  Failed to open port at {baud}")
            
    return None, None

def main():
    ports = get_available_ports()
    if not ports:
        print("No ports found (/dev/ttyUSB* or /dev/ttyACM*).")
        print("Check connection and permissions (sudo chmod 666 /dev/tty...).")
        return

    print(f"Found ports: {ports}")
    
    for port in ports:
        baud, ids = scan_port(port)
        if baud:
            print(f"\n[RESULT] Recommended Settings for {port}:")
            print(f"  BAUDRATE = {baud}")
            print(f"  IDs found = {ids}")
            return
            
    print("\n[RESULT] No motors found on any port/baudrate.")

if __name__ == "__main__":
    main()
