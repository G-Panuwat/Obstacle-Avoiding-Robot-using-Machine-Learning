import pygame
import math
import csv
import sys

# 1. Basic window and color settings
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("AI Rover - Data Collection Simulator")
clock = pygame.time.Clock()

# 2. Rover robot settings
robot_x, robot_y = 100, 500
robot_angle = 0
robot_speed = 2
turn_speed = 0.04
robot_radius = 15

# 3. Define all obstacles in the arena
obstacles = [
    # Outer Boundaries (walls surrounding the arena)
    pygame.Rect(0, 0, 800, 20),      # Top wall
    pygame.Rect(0, 580, 800, 20),    # Bottom wall
    pygame.Rect(0, 0, 20, 600),      # Left wall
    pygame.Rect(780, 0, 20, 600),    # Right wall

    # Top-Left Room
    pygame.Rect(20, 200, 150, 20),   # Bottom wall of the room
    pygame.Rect(170, 20, 20, 100),   # Right wall of the room (gap left at the bottom for a doorway)

    # Bottom-Right Room
    pygame.Rect(600, 400, 180, 20),  # Top wall of the room
    pygame.Rect(600, 400, 20, 100),  # Left wall of the room (gap left at the bottom for a doorway)

    # Center Obstacles (simulating tables or furniture)
    pygame.Rect(350, 250, 100, 80),  # Large obstacle in the center
    pygame.Rect(250, 400, 40, 40),   # Small pillar 1
    pygame.Rect(500, 150, 40, 40),   # Small pillar 2

    # Hallway Dividers
    pygame.Rect(320, 20, 20, 120),   # Wall protruding from the top
    pygame.Rect(460, 460, 20, 120)   # Wall protruding from the bottom
]

# 4. Sensor simulation function (Raycasting)
def cast_ray(start_x, start_y, angle):
    max_dist = 300  # Maximum sensor visibility range (in pixels)
    for dist in range(max_dist):
        target_x = start_x + math.cos(angle) * dist
        target_y = start_y + math.sin(angle) * dist
        
        # Ray hits the window boundary
        if target_x < 0 or target_x > WIDTH or target_y < 0 or target_y > HEIGHT:
            return dist
        
        # Ray hits an obstacle
        for obs in obstacles:
            if obs.collidepoint(target_x, target_y):
                return dist
    return max_dist

# Variables for storing the dataset
dataset = []
# Define the table header (Features and Label)
dataset.append(["sensor_left", "sensor_front", "sensor_right", "action"])

# 5. Main simulation loop
running = True
frame_count = 0

while running:
    screen.fill(WHITE)
    
    # Draw all obstacles
    for obs in obstacles:
        pygame.draw.rect(screen, BLACK, obs)
        
    # Check for window close event
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 6. Read keyboard input and assign action label
    keys = pygame.key.get_pressed()
    action = 0  # 0: Stop/Forward, 1: Turn Left, 2: Turn Right
    
    if keys[pygame.K_LEFT]:
        robot_angle -= turn_speed
        action = 1
    if keys[pygame.K_RIGHT]:
        robot_angle += turn_speed
        action = 2
    if keys[pygame.K_UP]:
        robot_x += math.cos(robot_angle) * robot_speed
        robot_y += math.sin(robot_angle) * robot_speed
        # If not turning left or right, treat as moving straight forward
        if action == 0:
            action = 0
            
    # Draw the robot as a circle
    pygame.draw.circle(screen, BLUE, (int(robot_x), int(robot_y)), robot_radius)

    # 7. Read sensor values from 3 directions (Features)
    # Left (-45 degrees), Front (0 degrees), Right (+45 degrees)
    dist_left = cast_ray(robot_x, robot_y, robot_angle - math.pi/4)
    dist_front = cast_ray(robot_x, robot_y, robot_angle)
    dist_right = cast_ray(robot_x, robot_y, robot_angle + math.pi/4)

    # Draw the sensor laser rays for visual feedback
    pygame.draw.line(screen, RED, (robot_x, robot_y), 
                     (robot_x + math.cos(robot_angle - math.pi/4)*dist_left, robot_y + math.sin(robot_angle - math.pi/4)*dist_left), 2)
    pygame.draw.line(screen, GREEN, (robot_x, robot_y), 
                     (robot_x + math.cos(robot_angle)*dist_front, robot_y + math.sin(robot_angle)*dist_front), 2)
    pygame.draw.line(screen, RED, (robot_x, robot_y), 
                     (robot_x + math.cos(robot_angle + math.pi/4)*dist_right, robot_y + math.sin(robot_angle + math.pi/4)*dist_right), 2)

    # 8. Log data every 5 frames to avoid overly redundant/repetitive samples
    frame_count += 1
    if frame_count % 5 == 0 and keys[pygame.K_UP]:  # Only record data when the robot is moving
        dataset.append([round(dist_left, 2), round(dist_front, 2), round(dist_right, 2), action])

    pygame.display.flip()
    clock.tick(60)  # Run at 60 FPS

# 9. Save the dataset as a CSV file when the program is closed
pygame.quit()
print(f"Collected {len(dataset)-1} samples.")

with open('rover_navigation_dataset.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(dataset)
    
print("Dataset saved to 'rover_navigation_dataset.csv'")
sys.exit()
