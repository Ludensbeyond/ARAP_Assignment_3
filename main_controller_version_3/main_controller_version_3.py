# main_controller script
import robot

def main():
    robot1 = robot.ARAP()
    robot1.init_devices()
    
    red_seen = False
    green_seen = False
    blue_seen = False
    
    while True:
        robot1.reset_actuator_values()
        robot1.get_sensor_input()
        robot1.blink_leds()
        
        colour = robot1.get_camera_image(5)      
            
        if colour == "red" and not red_seen:
            print(" ")
            print("Colour of Block Detected")
            print("I see red")
            print(" ")
            red_seen = True
        elif colour == "blue" and not blue_seen:
            print(" ")
            print("Colour of Block Detected")
            print("I see blue")
            print(" ")
            blue_seen = True
        elif colour == "green" and not green_seen:
            print(" ")
            print("Colour of Block Detected")
            print("I see green")
            print(" ")
            green_seen = True
        
        if red_seen and blue_seen and green_seen:
            print(" ")
            print("Summary:")
            print("All colors encountered")
            robot1.stop()  # Stop the robot            
            break
            
        if robot1.front_obstacles_detected():
            robot1.move_backward()
            robot1.turn_left()
        elif robot1.left_obstacles_detected():
            robot1.move_backward()
            robot1.turn_right()
        elif robot1.right_obstacles_detected():
            robot1.move_backward()
            robot1.turn_right()    
        else: 
            robot1.run_braitenberg()
        robot1.set_actuators()
        robot1.step()

if __name__ == "__main__":
    main()