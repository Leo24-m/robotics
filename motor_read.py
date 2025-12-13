import csv
import time
import os
import sys
import tty
import termios
from dynamixel_sdk import *  # Dynamixel SDK

# Set up parameters
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'

# Dynamixel IDs
DXL_ID_P2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Protocol 2.0
MX28_IDS = [0, 3, 6, 9]  # MX28 IDs
MX64_IDS = [1, 2, 4, 5, 7, 8, 10, 11]  # MX64 IDs

# Initial angles for the motors
INITIAL_ANGLES = {
    0: 180.0,  1: 180.0,  2: 180.0,  # Leg 1
    3: 180.0,  4: 180.0,  5: 180.0,  # Leg 2
    6: 180.0,  7: 180.0,  8: 180.0,  # Leg 3
    9: 180.0, 10: 180.0, 11: 180.0   # Leg 4
}

# Dynamixel setup
port_handler = PortHandler(DEVICENAME)
packet_handler = PacketHandler(2.0)

# Open port
if not port_handler.openPort():
    print("Failed to open the port.")
    sys.exit()
if not port_handler.setBaudRate(BAUDRATE):
    print("Failed to set the baudrate.")
    sys.exit()

# Function to read motor angle
def get_motor_angle(motor_id):
    # Get present position of the motor
    dxl_present_position, dxl_comm_result, dxl_error = packet_handler.read4ByteTxRx(port_handler, motor_id, 132)
    if dxl_comm_result != COMM_SUCCESS:
        print(f"Failed to get motor {motor_id} position")
        return None
    return dxl_present_position

# Function to set motor angle
def set_motor_angle(motor_id, angle):
    # Write goal position (angle) to the motor
    goal_position = int((angle / 360.0) * 4095.0)  # Converting angle to Dynamixel's position value
    dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(port_handler, motor_id, 116, goal_position)

    if dxl_comm_result != COMM_SUCCESS:
        print(f"[ID:{dxl_id}] Move Fail: {packet_handler.getTxRxResult(dxl_comm_result)}")
    elif dxl_error != 0:
        print(f"[ID:{dxl_id}] Move Fail: {packet_handler.getRxPacketError(dxl_error)}")

# Function to unlock torque for specified motors
def unlock_torque(motor_ids):
    for motor_id in motor_ids:
        packet_handler.write1ByteTxRx(port_handler, motor_id, 64, 0)  # Disable Torque
        print(f"Torque disabled for motor {motor_id}")

# Function to enable torque for specified motors
def enable_torque(motor_ids):
    for motor_id in motor_ids:
        packet_handler.write1ByteTxRx(port_handler, motor_id, 64, 1)  # Enable Torque
        print(f"Torque enabled for motor {motor_id}")

# Function to save angles to CSV file (append mode)
def save_angles_to_csv(angles):
    filename = "motor_angles.csv"
    # Check if the file exists
    file_exists = os.path.isfile(filename)
    
    # Open the CSV file in append mode
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # If the file doesn't exist, write the header first
        if not file_exists:
            writer.writerow(["9_motor", "10_motor", "11_motor"])
            # writer.writerow(["0_motor", "1_motor", "2_motor"])
        
        row = []
        for motor_id in [9, 10, 11]:
        # for motor_id in [0, 1, 2]:
            angle = angles.get(motor_id)
            row.append(angle if angle is not None else "N/A")
        writer.writerow(row)
    
    print(f"Angles saved to {filename}")

# Capture keypress for interactive control
def get_keypress():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

# Function to move all motors to their initial angles immediately (no interpolation)
def move_all_motors_to_initial():
    enable_torque(DXL_ID_P2)
    for motor_id in DXL_ID_P2:
        current_angle = get_motor_angle(motor_id)
        if current_angle is not None:
            target_angle = INITIAL_ANGLES[motor_id]
            # Directly move to the target angle
            set_motor_angle(motor_id, target_angle)
            print(f"Motor {motor_id} moved directly to target angle {target_angle}")

# Main loop for user interaction
def main():
    motor_ids_to_control = [9, 10, 11]  # Default motor IDs for control
    # motor_ids_to_control = [0, 1, 2]  # Changed to motor IDs 0, 1, 2 for control
    motor_angles = {}

    print("Press 'i' to move all motors to their initial positions (slowly) and enable torque.")
    print("Press 't' to disable torque for motors [3, 4, 5]")
    print("Press 'a' to save the current motor angles to a CSV file")
    print("Press 'q' to quit")

    while True:
        key = get_keypress()

        if key == 'i':
            print("Moving all motors to their initial positions...")
            move_all_motors_to_initial()
        elif key == 't':
            unlock_torque(motor_ids_to_control)
        elif key == 'a':
            for motor_id in motor_ids_to_control:
                angle = get_motor_angle(motor_id)
                if angle is not None:
                    motor_angles[motor_id] = angle
            if len(motor_angles) == 3:  # Save only if 3 motor angles are collected
                save_angles_to_csv(motor_angles)
                motor_angles.clear()  # Clear after saving
        elif key == 'q':
            print("Exiting...")
            break
        time.sleep(0.1)

    # Close the port after exiting
    port_handler.closePort()

if __name__ == "__main__":
    main()
