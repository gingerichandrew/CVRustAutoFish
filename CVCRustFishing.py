import mss
import os
import cv2
import math
import serial
import time
import win32gui, win32con, win32api, win32ui
import multiprocessing
import pandas as pd
import numpy as np
import pytesseract
import skimage
from skimage.metrics import structural_similarity as ssim
from skimage.exposure import match_histograms
from PIL import Image
from pynput import mouse
from random import *
from multiprocessing   import Process, Queue
from ctypes import wintypes, windll, create_unicode_buffer

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\andre\Tesseract-OCR\tesseract.exe'

# Player Cast rod, enable auto fish
# Real in fish
# Before recasting, check if the rod has bait
# If it doesn't have bait, open inventory and chop all unneeded fish
# If there is small trout, drag them on as bait
# If there is no small trout, use raw fish
# Put Salmon and Shark into chest
# Recast

def resize_screenshot(img, scale_percent):
    # Total compass w/h
    width = int(img.shape[1] * scale_percent / 100) 
    height = int(img.shape[0] * scale_percent / 100)

    # Dimensions to scale to
    dim = (width, height)

    # Resize img
    frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return frame

def screenshot(left, top, width, height):
    with mss.mss() as sct:
        # The screen part to capture
        monitor = {'top': top, 'left': left, 'width': width, 'height': height}
        # Get raw pixels from the screen, save it to a Numpy array
        img = np.array(sct.grab(monitor))
        return img


def get_config(): # Attempt to open config file
    try:
        config = open(r"C:/Program Files (x86)/Steam/steamapps/common/Rust/cfg/client.cfg", "r")
        config_content = config.readlines()
        # Loop through each line in client.cfg
        for line in config_content:
            # Input sensitivity
            if "input.sensitivity" in line:             
                SENSITIVITY = float(line.split('"')[1]) 
            # Field of view    
            if "graphics.fov" in line:                 
                FOV = float(line.split('"')[1])
            # Aim down sight sensitivity         
            if "input.ads_sensitivity" in line:         
                ADS_FACTOR = float(line.split('"')[1]) 
            # User interface scale 
            if "graphics.uiscale" in line:              
                UI_SCALE = float(line.split('"')[1])
    # Use default values if opening file failed    
    except:
        SENSITIVITY = 1 # Input sensitivity
        FOV = 90        # Field of view
        ADS_FACTOR = 1  # Aim down sight sensitivity
        UI_SCALE = 1    # User interface scale
        print("Failed to open CFG, using defaults")
    return SENSITIVITY, FOV, ADS_FACTOR, UI_SCALE


def mouse_move(queue_m):
    arduino = None
    # Attempt to setup serial communication over com-port
    try:
        arduino = serial.Serial('COM5', 115200)
        print('Arduino: Ready')
    except:
        print('Arduino: Could not open serial port - Now using virtual input')

    print("Mouse Process: Ready")
    while True:
        # Check if there is data waiting in the queue
        try:
            move_data = queue_m.get()
            out_x, out_y, click = move_data[0], move_data[1], move_data[2]
        except:
            print('Empty')
            continue
        # If using arduino, convert cooridnates
        if(arduino):
            if out_x < 0: 
                out_x = out_x + 256 
            if out_y < 0:
                out_y = out_y + 256 
            pax = [int(out_x), int(out_y), int(click)]
            # Send data to microcontroller to move mouse
            arduino.write(pax)          
        else:
            # Move mouse virtually
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, int(out_x), int(out_y), 0, 0)


def lerp(wt, ct, x1, y1, start, queue_m): # Linear interpolation
    x_, y_, t_ = 0, 0, 0
    for i in range(1, int(ct) + 1):
        xI = i * x1 // ct
        yI = i * y1 // ct
        tI = (i * ct) // ct
        # Put mouse input in queue
        queue_m.put([xI - x_, yI - y_, 0])
        sleep_time(tI - t_)
        x_ = xI
        y_ = yI
        t_ = tI
    # Find time remaining (Wait time - Control time loop)
    loop_time = (time.perf_counter() - start) * 1000
    sleep_time(wt - loop_time)


def sleep_time(wt): # More accurate sleep(Performance hit)
    target_time = time.perf_counter() + (wt / 1000)
    # Busy-wait loop until the current time is greater than or equal to the target time
    while time.perf_counter() < target_time:
        pass


def is_substring(arr, s):   # Finds if any string in an array of strings is a substring of a string
    for i in arr:
        if i in s:
            return True, i
    return False, None


def send_keystroke(key): # Send a keystroke, up and down
    win32api.keybd_event(key, 0, 0, 0)
    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)


def press_key(key): # Send a keystroke, down
    win32api.keybd_event(key, 0, 0, 0)


def release_key(key): # Send a keystroke, up
    win32api.keybd_event(key, 0, win32con.KEYEVENTF_KEYUP, 0)


def lift_mouse(queue_m): # Lifts left and right mouse, wait 50ms
    queue_m.put([0,0,6])
    sleep_time(50)


def get_inventory_slot(imgB, images_path, images):  # Compare image to set of images
    error_t = 0.0
    index = 0
    for i in range(len(images)):
        imgA = images[i].copy()
        imgB = imgB.copy()

        imgA = cv2.resize(imgA, (89, 89), interpolation = cv2.INTER_AREA)

        grayA = cv2.cvtColor(imgA, cv2.COLOR_RGB2GRAY)
        grayB = cv2.cvtColor(imgB, cv2.COLOR_RGB2GRAY)

        s = ssim(grayB, grayA)
        if(s > error_t):
            index = i
            error_t = s
    return images_path[index].split(".")[0]


def get_inventory(fish_images_path, fish_images): # Get values and coordinates for each slot
    # Will store all inventory values
    inventory = []
    # Number of slots = rows*columns
    rows, columns = 4, 6
    # Starting position / gap px length
    left, top, w, h, g = 660, 573, 89, 89, 7
    for r in range(rows):
        row = []
        for c in range(columns):
            # Calculate current slot position
            left_current, top_current, w_current, h_current = left + ((w + g) * c), top + ((h + g) * r), w, h
            # Take screenshot
            inventory_slot = screenshot(left_current, top_current, w_current, h_current)
            # Detect what object in slot
            slot_value = get_inventory_slot(inventory_slot, fish_images_path, fish_images)
            # Append value to row
            row.append([slot_value, left_current, top_current, w_current, h_current])
        # Append the entire row to inventory
        inventory.append(row)
    return inventory


def remove_bait(left, top, w, h):
    # Timing
    time_to_wait = 50
    time_to_move = 200

    # Coordiantes for where the rod in the inventory is
    rod_center_x, rod_center_y = int(left + w/2), int(top+h/2)
    # Coordinates for where the bait in the rod is
    bait_box_center_x, bait_box_center_y = 722, 324
    # Coordinates for where drop button is   
    drop_center_x, drop_center_y = 1112, 509

    # Get mouse position
    m_x, m_y = win32api.GetCursorPos()
    # Get delta from mouse to the rod
    dx_rod, dy_rod = rod_center_x - m_x, rod_center_y - m_y
    # Get delta from rod to bait box
    dx_bait_box, dy_bait_box = bait_box_center_x - rod_center_x , bait_box_center_y - rod_center_y
    # Get delta from bait box to rod
    dx_drop, dy_drop = drop_center_x - bait_box_center_x , drop_center_y - bait_box_center_y

    # First click on the rod that is broken, and has bait
    lerp(time_to_move + time_to_wait, time_to_move, dx_rod, dy_rod, time.perf_counter(), queue_m)        
    # Press left mouse                                                                              
    queue_m.put([0,0,2])
    # Wait
    sleep_time(randint(time_to_wait, time_to_wait * 2))
    # Release left mouse
    queue_m.put([0,0,3])

    # Wait 
    sleep_time(randint(time_to_wait, time_to_wait * 2))

    # Then move and right click on the bait
    lerp(time_to_move + time_to_wait, time_to_move, dx_bait_box, dy_bait_box, time.perf_counter(), queue_m)        
    # Press right mouse                                                                              
    queue_m.put([0,0,4])
    # Wait
    sleep_time(randint(time_to_wait, time_to_wait * 2))
    # Release right mouse
    queue_m.put([0,0,5])

    # Wait 
    sleep_time(350)

    # Drop the rod
    lerp(time_to_move + time_to_wait, time_to_move, dx_drop, dy_drop, time.perf_counter(), queue_m)        
    # Press left mouse                                                                              
    queue_m.put([0,0,2])
    # Wait
    sleep_time(randint(time_to_wait, time_to_wait * 2))
    # Release left mouse
    queue_m.put([0,0,3])

    sleep_time(350)
    print("REMOVING BAIT AND DROPPING")


def get_rod_health(hotbar_): # Get the health of the fishing rod
    hotbar_slot = resize_screenshot(hotbar_, 200)
    # Get width and height from shape of image
    w, h = hotbar_slot.shape[1], hotbar_slot.shape[0]
    # Var to store health in percentage
    rod_health = 0
    # Copy health bar from image 
    health_bar = hotbar_slot[0:h, 0:5].copy()
    # Convert from BGRA to BGR
    health_bar = cv2.cvtColor(health_bar, cv2.COLOR_BGRA2BGR)
    # Convert image to hsv
    health_bar_hsv = cv2.cvtColor(health_bar, cv2.COLOR_BGR2HSV)
    # Create green mask 
    green_mask = cv2.inRange(health_bar, np.array([36, 25, 25]), np.array([70, 255,255]))
    # Get bitwise and frame - mask
    health_bar = cv2.bitwise_and(health_bar_hsv, health_bar_hsv, mask=green_mask)
    # Convert rod to grayscale
    health_bar = cv2.cvtColor(health_bar, cv2.COLOR_BGRA2GRAY)

    # Find health
    for i in range(h):
        if(health_bar[i, 2] > 100 and health_bar[i, 2] < 130):
            # Calculate health, percentage of pixels that are green
            rod_health = (100 - (math.floor((((i + 1)/h) * 100)))) + 1
            return rod_health


def get_rod_bait(hotbar_slot): # Check if the rod currently has bait or not
    # Get width and height from shape of image
    w, h = hotbar_slot.shape[1], hotbar_slot.shape[0]
    # Copy bait box from image 
    rod_bait_box = hotbar_slot[6:11, 9:14].copy()
    # Convert from BGRA to BGR
    rod_bait_box = cv2.cvtColor(rod_bait_box, cv2.COLOR_BGRA2BGR)
    # Convert bait box to grayscale
    rod_bait_box = cv2.cvtColor(rod_bait_box, cv2.COLOR_BGRA2GRAY)
    # Get sum of black and white pixels
    black_pxs = np.sum(rod_bait_box < 200)
    white_pxs = np.sum(rod_bait_box >= 200)
    # If there is more white pixels then black, return there is bait
    if(white_pxs > black_pxs):
        return True
    # Else the rod doesn't have bait
    return False


def get_hotbar(): # Find where fishing rod is positioned
    # Will store hotbar results
    hotbar = []
    # Image paths
    hotbar_path = ["rod.jpg", "blank.jpg"]
    # Load images
    hotbar_images = [cv2.imread(f'./fish_images/{hotbar_path[i]}') for i in range(len(hotbar_path))]
    # Number of columns
    columns = 6
    # Starting coordinate position
    left, top, w, h, g = 661, 963, 89, 89, 7
    for c in range(columns):
        # Get current coordinates
        left_current, top_current = left + ((w + g) * c), top
        # Get screenshot
        hotbar_slot = screenshot(left_current, top_current, w, h)
        # Get detection results
        slot_value = get_inventory_slot(hotbar_slot, hotbar_path, hotbar_images)
        if("rod" in slot_value):
            # Get the rods health
            rod_health = get_rod_health(hotbar_slot)
            # Get its bait status
            has_bait = get_rod_bait(hotbar_slot)
            # Append the column to the hotbar
            hotbar.append([left_current, top_current, w, h, rod_health, has_bait])
            # Return the rod if its not broken
            if (rod_health != 0 and rod_health is not None):
                print(rod_health)
                return left_current, top_current, w, h, rod_health, has_bait, c
            # If rod is broken, and has bait
            elif(rod_health is None and has_bait):
                print(rod_health)
                remove_bait(left_current, top_current, w, h)
    return -1, -1, -1, -1, -1, -1, -1


def drag_and_drop(x1, y1, x2, y2, queue_m): # Move mouse to x1y1 and drag and drop at x2y2
    # Timing
    time_to_wait = 50
    time_to_move = 200
    # Lift left and right mouse button
    lift_mouse(queue_m)
    # Get mouse cursor position
    m_x, m_y = win32api.GetCursorPos()
    # Get delta from mouse to the item
    dx1, dy1 = x1 - m_x , y1 - m_y 
    # Get delta from item to destination
    dx2, dy2 = x2 - x1 , y2 - y1
    # Move mouse to item
    lerp(time_to_move + time_to_wait, time_to_move, dx1, dy1, time.perf_counter(), queue_m)         # 250ms
    # Put left mouse press in queue
    queue_m.put([0,0,2])
    # Wait before moving
    sleep_time(time_to_wait)                                                                        # 50ms
    # Drag item to destination
    lerp(time_to_move + time_to_wait, time_to_move,  dx2, dy2, time.perf_counter(), queue_m)        # 250ms
    # Wait then release
    sleep_time(time_to_wait)                                                                        # 50ms
    # Put left mouse release in queue
    queue_m.put([0,0,3])
    # Wait for mouse to release
    sleep_time(time_to_wait)                                                                        # 50ms
    # Return true so function calling can break out
    return True 


def restock_rod(fish_images_path, fish_images, rod_x, rod_y, queue_m): # Attempt to restock the fishing rod
    # Get inventory
    inventory = get_inventory(fish_images_path, fish_images)
    lift_mouse(queue_m)
    # List to store possible trout
    trout_coordinates = []
    # Raw coordinates to store possible raw fish
    raw_coordinates = []
    # Number of rows/column in inventory
    rows, columns = 4, 6
    # Starting position / gap px length
    left, top, w, h, g = 660, 573, 89, 89, 7
    # Loop through the colums and rows in inventory
    for r in range(rows):
        for c in range(columns):
            # Get item, and coordinates
            slot_value, left_current, top_current, w, h = inventory[r][c]
            # If trout, or if raw fish, add them to each storage list
            if(slot_value == "trout"): #
                trout_x, trout_y = int(left_current + w/2), int(top_current+h/2)
                trout_coordinates.append([trout_x, trout_y])
            elif(slot_value == "raw"):
                raw_x, raw_y = int(left_current + w/2), int(top_current+h/2)
                raw_coordinates.append([raw_x, raw_y])
    # Prioritize using the trout as bait before raw fish 
    if(len(trout_coordinates) > 0):
        # Draw trout onto rod
        return drag_and_drop(trout_coordinates[0][0], trout_coordinates[0][1], rod_x, rod_y, queue_m)
    if(len(raw_coordinates) > 0):
        # Draw raw fish onto rod
        return drag_and_drop(raw_coordinates[0][0], raw_coordinates[0][1], rod_x, rod_y, queue_m)
    # Return false if no bait to be used
    return False

def find_buttons():
    # Sleep
    sleep_time(500)
    # List of possible fish
    fish_list = ['ANCHOVY', 'CATFISH', 'HERRING','MINNOW','ROUGHLY','SALMON','SARDINE','SHARK','TROUT', 'PERCH']
    # The two possible locations for "gut", "drop"
    gut_p = [[993, 440, 239, 51], [993, 352, 239, 51]]
    drop_p = [[993, 484, 239, 51], [993, 406, 239, 51]]
    name_p = [663,107, 567, 30]

    x_n, y_n, w_n, h_n = name_p[0], name_p[1], name_p[2], name_p[3]
    name_img = screenshot(x_n, y_n, w_n, h_n)
    name_img = resize_screenshot(name_img, 200)
    name_img = cv2.cvtColor(name_img, cv2.COLOR_BGRA2GRAY) # Convert name_img to grayscale
    name_img = cv2.bitwise_not(name_img)
    name_text = pytesseract.image_to_string(name_img)

    is_fish, fish_name = is_substring(fish_list, name_text)
    if(is_fish == False):
        return -1, -1
    # Look at first position, then if needed the next position
    for i in range(len(gut_p)):
        # Get gut screenshot
        x_g, y_g, w_g, h_g = gut_p[i][0], gut_p[i][1], gut_p[i][2], gut_p[i][3]
        gut_img = screenshot(x_g, y_g, w_g, h_g)
        gut_img = cv2.cvtColor(gut_img, cv2.COLOR_BGRA2GRAY) # Convert gut_img to grayscale
        gut_img = cv2.bitwise_not(gut_img)
        # Get drop screenshot
        x_d, y_d, w_d, h_d = drop_p[i][0], drop_p[i][1], drop_p[i][2], drop_p[i][3]
        drop_img = screenshot(x_d, y_d, w_d, h_d)   
        drop_img = cv2.cvtColor(drop_img, cv2.COLOR_BGRA2GRAY) # Convert drop_img to grayscale
        drop_img = cv2.bitwise_not(drop_img)
        # Get text of both screenshots
        gut_text = pytesseract.image_to_string(gut_img)
        drop_text = pytesseract.image_to_string(drop_img)
        # Check if gut image containts "gut", and drop image contains "drop"
        if("Gut" in gut_text or "Drop" in drop_text):
            # Return positions of both
            return [x_g, y_g, w_g, h_g], [x_d, y_d, w_d, h_d]
    # If not found, return -1 for both
    return -1, -1

def gut_fish(fish_images_path, fish_images, strict, queue_m): # Gut all the fish in inventory that aren't sharks, trout, or salmon
    # Fish to gut
    fish_list = ['anchovy', 'catfish', 'herring','minnow','roughly','sardine','perch']
    # Wait time variables
    time_to_wait = 50
    time_to_move = 200
    # If strict false, append salmon
    only_gut_once = False
    if(not strict):
        only_gut_once = True
        fish_list.append('salmon')
    # Number of rows/column in inventory
    rows, columns = 4, 6
    # Starting position / gap px length
    left, top, w, h, g = 660, 573, 89, 89, 7
    # Reset Mouse
    lift_mouse(queue_m)
    # Get inventory
    inventory = get_inventory(fish_images_path, fish_images)
    # Loop through the colums and rows in inventory
    for r in range(rows):
        for c in range(columns):
            # Get item, and coordinates
            slot_value, left_current, top_current, w, h = inventory[r][c]
            # Calc item center
            item_center_x, item_center_y = (left_current + w/2), (top_current + h/2)
            # If shark, trout, salmon, ignore(unless strict on)
            if(slot_value not in fish_list):    
                continue

            # Get mouse position
            m_x, m_y = win32api.GetCursorPos()
            # Get deltas
            dx, dy = item_center_x - m_x, item_center_y - m_y
            # Move mouse to item
            lerp(time_to_move + time_to_wait, time_to_move, dx, dy, time.perf_counter(), queue_m)

            # Click on item
            queue_m.put([0,0,2])
            # Sleep for 50ms
            sleep_time(time_to_wait + time_to_wait)
            # release on item
            queue_m.put([0,0,3])

            # Find location of "Gut" and "Drop"
            gut_coordinates, drop_coordinates = find_buttons()
            # While buttons are present
            while(gut_coordinates != -1 or drop_coordinates != -1):
                sleep_time(100)
                # Calc center coordinates
                gut_center_x, gut_center_y = (gut_coordinates[0] + gut_coordinates[2]/2), (gut_coordinates[1] + gut_coordinates[3]/2)
                # Get mouse position
                m_x, m_y = win32api.GetCursorPos()
                # Get deltas
                dx, dy = gut_center_x - m_x, gut_center_y - m_y
                # Move mouse to options
                lerp(time_to_move + time_to_wait, time_to_move, dx, dy, time.perf_counter(), queue_m)

                # Click on item
                queue_m.put([0,0,2])
                # Sleep for 50ms
                sleep_time(time_to_wait + time_to_wait)
                # release on item
                queue_m.put([0,0,3])

                # Sleep for 100ms
                sleep_time(time_to_wait)

                # Get next loop
                gut_coordinates, drop_coordinates = find_buttons()
                if(only_gut_once):
                    return


def cast_fishingrod(queue_m, previous_rod,fish_images_path, fish_images): # Cast the fishing rod
    # So that rod can be recast faster
    timer = time.perf_counter()
    # Rod casting timing
    min_rod_reset, max_rod_reset = 6500, 6750
    min_wait_cast, max_wait_cast = 1000, 1150
    min_cast_hold, max_cast_hold = 250, 350
    min_release_right, max_release_right = 500, 650
    # Reset Mouse
    lift_mouse(queue_m)
    # Press tab and open inventory
    send_keystroke(0x09)
    # Wait for inventory to be open
    sleep_time(300)
    # Get hotbar, returns paosition of sihing rod, health, and if it has bait 
    left, top, w, h, health, has_bait, c = get_hotbar()
    # If no rod was found
    if(c == -1):
        return -1

    # If the current rod doesn't have bait
    if(not has_bait):
        # Calc rod center position
        rod_center_x, rod_center_y = int(left + w/2), int(top+h/2)
        # Then gut fish
        gut_fish(fish_images_path, fish_images, True, queue_m)
        # Attempt to restock fishing rod
        success = restock_rod(fish_images_path, fish_images, rod_center_x, rod_center_y, queue_m)
        # If it wasn't able to, lack of bait
        if not success:
            print("No raw fish / small fish to gut - Adding Salmon")
            # Then gut fish including salmon
            gut_fish(fish_images_path, fish_images, False, queue_m)

            # Attempt to restock fishing rod again
            success_with_salmon = restock_rod(fish_images_path, fish_images, rod_center_x, rod_center_y, queue_m)
            if not success_with_salmon:
                print("COULDNT RESTOCK - ADD SHARK")

    # If the previous rod was broken
    if(c != previous_rod):
        print("Rod Broke - changing to new rod: ", previous_rod, " -> ", c)
        # Send keystroke to press on rod
        press_key(0x30 + (c+1))
        # Wait for rod to be pressed
        sleep_time(randint(50,60))
        # Send keystroke to release rod
        release_key(0x30 + c)
        # Wait for rod to be pressed
        sleep_time(randint(50,60))

    # Close inventory
    send_keystroke(0x09)

    # Reset Mouse
    lift_mouse(queue_m)
    # Get time remaining for us to recast
    time_remaining = min_rod_reset - ((time.perf_counter() - timer) * 1000)
    if(time_remaining > 0):
        # Sleep for remaining time out of the 6.5 seconds
        sleep_time(time_remaining)

    # Put right mouse press in queue
    queue_m.put([0,0,4])

    # Sleep for a second to a second and a .250ms to wait for to be ready to cast
    sleep_time(randint(min_wait_cast, max_wait_cast))
    # Put left press in queue
    queue_m.put([0,0,2])

    # Sleep for 500ms to 750ms before releasing left press
    sleep_time(randint(min_cast_hold, max_cast_hold))
    # Put left mouse release in queue
    queue_m.put([0,0,3])

    # Sleep for 1 to 2 seconds, before releasing right 
    sleep_time(randint(min_release_right, max_release_right))
    # Put right mouse release in queue
    queue_m.put([0,0,5])

    return c


def detect_caught_fish(queue_f): # Detect if a fish has been caught using pytesseract
    # List of possible fish
    fish_list = ['ANCHOVY', 'CATFISH', 'HERRING','MINNOW','ROUGHLY','SALMON','SARDINE','SHARK','TROUT', 'PERCH']
    # Overall timer
    ov_timer = time.perf_counter()
    # Time stamp, used for checking if text should be checked for fish
    start_timer = time.perf_counter()
    # Caught Demensions
    x, y, w, h = 1607, 724, 287, 80
    while True:
        c_time = time.perf_counter()
        # Take screenshot to check if something has been caught every second
        if(((c_time - start_timer) * 1000) > 500):
            # Take Screenshot
            caught_img = screenshot(x, y, w, h)
            caught_img = resize_screenshot(caught_img, 200)
            caught_img = cv2.cvtColor(caught_img, cv2.COLOR_BGRA2GRAY) # Convert name_img to grayscale
            caught_img = cv2.bitwise_not(caught_img)
            ret,caught_img = cv2.threshold(caught_img,127,255,cv2.THRESH_BINARY)
            # Check if there is text in the screenshot
            fish = pytesseract.image_to_string(caught_img)
            
            cv2.imshow("Fish Cam", caught_img)
            # waits for user to press any key
            # (this is necessary to avoid Python kernel form crashing)
            cv2.waitKey(27)
            
            print(round((c_time - ov_timer), 2), " seconds | ", round((c_time - ov_timer)/60, 2), " minutes", " | ", fish)
            if len(fish) != 0:
                # Check if the string found is a fish from the list, and get its name
                fish_detected, fish_name = is_substring(fish_list, fish)
                # If there is a fish in the string, put the fish in the queue
                if(fish_detected):
                    queue_f.put(fish_name)
                    # Then sleep for 7 seconds to ensure its not double counted
                    sleep_time(6000)
            start_timer = time.perf_counter()
    # closing all open windows
    cv2.destroyAllWindows()


def fish(queue_m, queue_f): # Fishing main loop
    # Image paths
    fish_images_path = ["anchovy.jpg","herring.jpg","raw.jpg","rod.jpg","salmon.jpg","sardine.jpg","shark.jpg","trout.jpg", "blank.jpg"]
    # Load comparison images
    fish_images = [cv2.imread(f'./fish_images/{fish_images_path[i]}') for i in range(len(fish_images_path))]
    # Dictionary to store how many of each fish you've caught
    fish_caught = {
        "ANCHOVY": {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "value": 0
        },
        "HERRING": {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "value": 0
        },
        "SARDINE": {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "value": 0
        },
        "TROUT": {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "value": 0
        },
        "SALMON": {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "value": 22.5
        },
        "SHARK": {
            "count": 0,
            "total_time": 0,
            "avg_time": 0,
            "value": 45
        }
    }
    # Press / Release Timing
    min_press, max_press = 390, 410
    min_release, max_release = 390, 410
    # Auto fishing toggle
    fishing = False
    press = False
    release = False
    key_down = False
    key = 0x00
    # Time stamp, used for checking if mouse input should be sent
    start_timer = time.perf_counter()
    # Time stamp, used for checking total time to catch a fish
    fish_timer = time.perf_counter()
    # Store total number of time per fish, to find average fish time, and total scrap count
    sum_amount_time = 0
    sum_amount_scrap = 0
    # Amount of time inbetween mouse input for first iteration
    time_to_sleep = 5000
    # Variable to track if rod needs to be cast again
    cast_rod = False
    # Variable to track if rod changed
    previous_rod = 0
    # Track current number of fish
    fish_count = 0
    # Main Loop
    print("Fishing Process: Ready")
    while True:
        # Check DOWN arrow key
        if(win32api.GetAsyncKeyState(0x28) < 0):
            while(win32api.GetAsyncKeyState(0x28) < 0):
                pass
            if(fishing):
                fishing = False
                lift_mouse(queue_m)
                # Stop moving
                if(key_down):
                    release_key(key)
                    key_down, key = False, 0x00
            else:
                fishing = True
                fish_timer = time.perf_counter()
            print("Auto Fishing: ", fishing)

        # Check UP arrow
        if(win32api.GetAsyncKeyState(0x26) < 0):
            while(win32api.GetAsyncKeyState(0x26) < 0):
                pass
            cast_rod = True

        # If auto fishing is not enabled, start loop from the top
        if not fishing:
            continue
        try:
            # Get fish that was caught
            fish = queue_f.get(block = False)
            # Stop moving
            if(key_down):
                release_key(key)
                key_down, key = False, 0x00
            # Time it took to catch the fish
            time_fish_took = math.floor((time.perf_counter() - fish_timer))
            # Add fish count to dictionary
            fish_caught[fish]["count"] += 1
            # Add it to the total time for specifc fish
            fish_caught[fish]["total_time"] += time_fish_took
            # Set specific fish average
            fish_caught[fish]["avg_time"] = math.floor((fish_caught[fish]["total_time"] / fish_caught[fish]["count"]))
            # Add it to the total time for all fishes
            sum_amount_time += time_fish_took
            # Add to scrap count
            sum_amount_scrap += fish_caught[fish]["value"]
            # Add to fish count
            fish_count += 1
            # sum_amount_time(s)
            scrap_per_hour = 0
            if(sum_amount_scrap != 0):
                scrap_per_hour = math.floor((((sum_amount_scrap)/sum_amount_time) * 60) * 60)
            # Rod needs to be recast
            cast_rod = True
            print("Type of fish:        ", fish)
            print("Time to catch:       ", time_fish_took)
            print("Avg catch time:      ", sum_amount_time/fish_count)
            print("Fishing for:         ", round(sum_amount_time/60, 2), "minutes")
            print("Total fish caught:   ", fish_count)
            print("Total scrap value:   ", sum_amount_scrap)
            print("Scrap per hour:      ", scrap_per_hour)
            print("Fish stats:          ", fish_caught)
        except:
            pass        # No new fish caught
        # If rod needs to be cast or its been longer than 1 minute 30 without catching a fish
        if(cast_rod or (time.perf_counter() - fish_timer) > 90):
            # Restart cast
            cast_rod = False
            press, release = False, False
            if(key_down):
                release_key(key)
                key_down, key = False, 0x00
            # Reset mouse left and right
            queue_m.put([0,0,6])
            # Sleep for delay
            sleep_time(500)
            # Cast rod
            previous_rod = cast_fishingrod(queue_m, previous_rod, fish_images_path, fish_images)
            # If no rod could be found
            if(previous_rod == -1):
                print("OUT OF RODS: EXIT")
                break
            # Set start timer so left presses dont begin in an insant
            start_timer, fish_timer = time.perf_counter(), time.perf_counter()

        if(((time.perf_counter() - start_timer) * 1000) > time_to_sleep):
            # If mouse should be pressed
            if(press):
                # Put left mouse press in queue
                queue_m.put([0,0,2])
                press, release = False, True
                start_timer = time.perf_counter()
                time_to_sleep = randint(min_press, max_press)
                # 30% chance for a key board press
                if(randint(0, math.floor(time.perf_counter() - fish_timer)) < 10):
                    if(randint(0,10) >= 5):
                        key_down, key = True, 0x41 # A
                        press_key(0x41)
                    else:
                        key_down, key = True, 0x44 # D
                        press_key(0x44)
                continue

            # If mouse should be released
            if(release):
                # Put left mouse release in queue
                queue_m.put([0,0,3])
                press, release = True, False
                start_timer = time.perf_counter()
                time_to_sleep = randint(min_release, max_release)
                # If key is down, release the key
                if(key_down):
                    release_key(key)
                    key_down, key = False, 0x00
                continue

            # Put left mouse press in queue
            queue_m.put([0,0,2])
            press, release = False, True
            start_timer = time.perf_counter()
            time_to_sleep = randint(min_press, max_press)



if __name__ == '__main__':
    multiprocessing.freeze_support()

    # Queue to communicate mouse movements, Fish -> queue_m -> Mouse
    queue_m = Queue(1)
    # Queue to communicate fish detection, Detect Fish -> queue_f -> Fish
    queue_f = Queue(1)

    # Start Mouse process, handles sending mouse out data, communicate with using queue_m
    mouse = Process(target=mouse_move, args=(queue_m,))
    mouse.daemon = True
    mouse.start()
    print("Mouse PID:", mouse.pid)

    # Start Mouse process, handles sending mouse out data, communicate with using queue_m
    detect_fish = Process(target=detect_caught_fish, args=(queue_f,))
    detect_fish.daemon = True
    detect_fish.start()
    print("Detect Fish PID:", detect_fish.pid)

    # Main process essentially becomes fishing, handles computing mouse movement coordinates
    print("Main PID:", os.getpid())
    fish(queue_m, queue_f)
    
