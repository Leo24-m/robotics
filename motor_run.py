import csv
import time
import os
from dynamixel_sdk import *  # Dynamixel SDK

# Set up parameters
BAUDRATE = 57600
DEVICENAME = '/dev/ttyUSB0'

# Dynamixel IDs
DXL_ID_P2 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Protocol 2.0

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

# Function to set motor position (no conversion needed since we use the position value directly)
def set_motor_position(motor_id, position):
    # Directly use the position value from CSV (No conversion to angle)
    print(f"Setting motor {motor_id} to position {position}")
    
    dxl_comm_result, dxl_error = packet_handler.write4ByteTxRx(port_handler, motor_id, 116, position)

    if dxl_comm_result != COMM_SUCCESS:
        print(f"Failed to set motor {motor_id} position")
    elif dxl_error != 0:
        print(f"Motor {motor_id} failed with error: {packet_handler.getRxPacketError(dxl_error)}")

# Function to move motors based on the CSV data within a specified duration
def move_motors_from_csv(csv_filename, duration=2.0):
    # Read the CSV file
    motor_data = []
    with open(csv_filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header row
        for row in reader:
            print(f"Reading row: {row}")  # Debugging: Show each row
            # Ensure that the row contains enough data (3 motor values)
            if len(row) >= 3:
                try:
                    # Try to convert the values to integers and append to motor_data
                    motor_data.append([int(row[0]), int(row[1]), int(row[2])])  # Add the motor positions
                except ValueError as e:
                    print(f"Skipping invalid row: {row}. Error: {e}")  # Skip invalid rows
            else:
                print(f"Skipping row with insufficient data: {row}")  # Skip rows with insufficient data

    # Debugging: Print the motor data
    print(f"Motor data collected: {motor_data}")

    if not motor_data:
        print("No valid motor data found. Exiting...")
        return

    # Enable torque for all motors before moving them
    # enable_torque([0, 1, 2])
    enable_torque([3, 4, 5])
    # enable_torque([6, 7, 8])
    # enable_torque([9, 10, 11])

    # Loop through each row of motor data
    for angles in motor_data:
        target_3_motor, target_4_motor, target_5_motor = angles
        print(f"Moving motors to positions: {target_3_motor}, {target_4_motor}, {target_5_motor}")

        # Gradually move all motors to the target positions synchronously
        current_positions = {
            # 0: get_motor_angle(0),
            # 1: get_motor_angle(1),
            # 2: get_motor_angle(2),

            3: get_motor_angle(3),
            4: get_motor_angle(4),
            5: get_motor_angle(5),

            # 6: get_motor_angle(6),
            # 7: get_motor_angle(7),
            # 8: get_motor_angle(8),

            # 9: get_motor_angle(9),
            # 10: get_motor_angle(10),
            # 11: get_motor_angle(11),
        }

        # Wait for the current positions to be retrieved
        if any(pos is None for pos in current_positions.values()):
            print("Error: Unable to get current positions. Skipping movement.")
            continue

        # Calculate step time to move to the target angle within the given duration
        steps = 10  # Number of steps for interpolation
        step_time = duration / steps

        # Gradually move motors synchronously
        for step in range(steps):
            # for motor_id, target_position in zip([0, 1, 2], [target_3_motor, target_4_motor, target_5_motor]):
            for motor_id, target_position in zip([3, 4, 5], [target_3_motor, target_4_motor, target_5_motor]):
            # for motor_id, target_position in zip([6, 7, 8], [target_3_motor, target_4_motor, target_5_motor]):
            # for motor_id, target_position in zip([9, 10, 11], [target_3_motor, target_4_motor, target_5_motor]):
                # Interpolate the position for each motor
                current_positions[motor_id] += (target_position - current_positions[motor_id]) / (steps - step)
                set_motor_position(motor_id, int(current_positions[motor_id]))

            time.sleep(step_time)

        # Final setting of target positions
        # set_motor_position(0, target_3_motor)
        # set_motor_position(1, target_4_motor)
        # set_motor_position(2, target_5_motor)

        set_motor_position(3, target_3_motor)
        set_motor_position(4, target_4_motor)
        set_motor_position(5, target_5_motor)

        # set_motor_position(6, target_3_motor)
        # set_motor_position(7, target_4_motor)
        # set_motor_position(8, target_5_motor)

        # set_motor_position(9, target_3_motor)
        # set_motor_position(10, target_4_motor)
        # set_motor_position(11, target_5_motor)

        print("Motors moved successfully.")
        time.sleep(0.1)  # Wait before starting the next movement

    # Disable torque for all motors after moving them
    # unlock_torque([0, 1, 2])
    unlock_torque([3, 4, 5])
    # unlock_torque([6, 7, 8])
    # unlock_torque([9, 10, 11])

# Main loop for user interaction
def main():
    print("Press 'q' to quit")

    # Specify the CSV file
    # csv_filename = '4_leg_motor_angles.csv'     # 1번, 4번 다리
    csv_filename = '2_leg_motor_angles.csv'     # 2번, 3번 다리
    duration = 0.0001  # Time in seconds for each motor movement

    while True:
        key = input("Press 'q' to quit, or press Enter to move motors based on CSV file: ")
        if key == 'q':
            print("Exiting...")
            break
        else:
            move_motors_from_csv(csv_filename, duration)

    # Close the port after exiting
    port_handler.closePort()

if __name__ == "__main__":
    main()
