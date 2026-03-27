import pygame
import math
import csv
import sys

# 1. ตั้งค่าพื้นฐานของหน้าต่างและสี
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

# 2. ตั้งค่าหุ่นยนต์ Rover
robot_x, robot_y = 100, 500
robot_angle = 0
robot_speed = 2
turn_speed = 0.04
robot_radius = 15


obstacles = [
    # กำแพงรอบนอก (Outer Boundaries)
    pygame.Rect(0, 0, 800, 20),      # กำแพงบน
    pygame.Rect(0, 580, 800, 20),    # กำแพงล่าง
    pygame.Rect(0, 0, 20, 600),      # กำแพงซ้าย
    pygame.Rect(780, 0, 20, 600),    # กำแพงขวา

    # ห้องมุมซ้ายบน (Top-Left Room)
    pygame.Rect(20, 200, 150, 20),   # กำแพงล่างของห้อง
    pygame.Rect(170, 20, 20, 100),   # กำแพงขวาของห้อง (เว้นช่องประตูไว้ด้านล่าง)

    # ห้องมุมขวาล่าง (Bottom-Right Room)
    pygame.Rect(600, 400, 180, 20),  # กำแพงบนของห้อง
    pygame.Rect(600, 400, 20, 100),  # กำแพงซ้ายของห้อง (เว้นช่องประตูไว้ด้านล่าง)

    # สิ่งกีดขวางตรงกลาง (จำลองโต๊ะหรือเฟอร์นิเจอร์)
    pygame.Rect(350, 250, 100, 80),  # สิ่งกีดขวางขนาดใหญ่ตรงกลาง
    pygame.Rect(250, 400, 40, 40),   # เสาเล็ก 1
    pygame.Rect(500, 150, 40, 40),   # เสาเล็ก 2

    # ผนังกั้นทางเดิน (Hallway Dividers)
    pygame.Rect(320, 20, 20, 120),   # กำแพงยื่นจากด้านบน
    pygame.Rect(460, 460, 20, 120)   # กำแพงยื่นจากด้านล่าง
]

# 4. ฟังก์ชันจำลองเซนเซอร์ (Raycasting)
def cast_ray(start_x, start_y, angle):
    max_dist = 300 # ระยะมองเห็นสูงสุดของเซนเซอร์
    for dist in range(max_dist):
        target_x = start_x + math.cos(angle) * dist
        target_y = start_y + math.sin(angle) * dist
        
        # ชนขอบหน้าต่าง
        if target_x < 0 or target_x > WIDTH or target_y < 0 or target_y > HEIGHT:
            return dist
        
        # ชนสิ่งกีดขวาง
        for obs in obstacles:
            if obs.collidepoint(target_x, target_y):
                return dist
    return max_dist

# ตัวแปรสำหรับเก็บข้อมูล Dataset
dataset = []
# กำหนดหัวตาราง (Features และ Label)
dataset.append(["sensor_left", "sensor_front", "sensor_right", "action"])

# 5. ลูปหลักของการจำลอง
running = True
frame_count = 0

while running:
    screen.fill(WHITE)
    
    # วาดสิ่งกีดขวาง
    for obs in obstacles:
        pygame.draw.rect(screen, BLACK, obs)
        
    # เช็ค Event การปิดหน้าต่าง
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # 6. รับคำสั่งจากคีย์บอร์ด (Label)
    keys = pygame.key.get_pressed()
    action = 0  # 0: หยุด/ตรงไป, 1: เลี้ยวซ้าย, 2: เลี้ยวขวา
    
    if keys[pygame.K_LEFT]:
        robot_angle -= turn_speed
        action = 1
    if keys[pygame.K_RIGHT]:
        robot_angle += turn_speed
        action = 2
    if keys[pygame.K_UP]:
        robot_x += math.cos(robot_angle) * robot_speed
        robot_y += math.sin(robot_angle) * robot_speed
        # ถ้าไม่ได้เลี้ยวซ้ายหรือขวา ให้มองว่าเดินหน้า
        if action == 0: 
            action = 0 
            
    # วาดหุ่นยนต์
    pygame.draw.circle(screen, BLUE, (int(robot_x), int(robot_y)), robot_radius)

    # 7. อ่านค่าเซนเซอร์ 3 ทิศทาง (Features)
    # ซ้าย (-45 องศา), หน้า (0 องศา), ขวา (+45 องศา)
    dist_left = cast_ray(robot_x, robot_y, robot_angle - math.pi/4)
    dist_front = cast_ray(robot_x, robot_y, robot_angle)
    dist_right = cast_ray(robot_x, robot_y, robot_angle + math.pi/4)

    # วาดเส้นเลเซอร์เซนเซอร์ให้เห็นภาพ
    pygame.draw.line(screen, RED, (robot_x, robot_y), 
                     (robot_x + math.cos(robot_angle - math.pi/4)*dist_left, robot_y + math.sin(robot_angle - math.pi/4)*dist_left), 2)
    pygame.draw.line(screen, GREEN, (robot_x, robot_y), 
                     (robot_x + math.cos(robot_angle)*dist_front, robot_y + math.sin(robot_angle)*dist_front), 2)
    pygame.draw.line(screen, RED, (robot_x, robot_y), 
                     (robot_x + math.cos(robot_angle + math.pi/4)*dist_right, robot_y + math.sin(robot_angle + math.pi/4)*dist_right), 2)

    # 8. บันทึกข้อมูล (เก็บทุกๆ 5 เฟรม เพื่อไม่ให้ข้อมูลซ้ำซากเกินไป)
    frame_count += 1
    if frame_count % 5 == 0 and keys[pygame.K_UP]: # เก็บข้อมูลเฉพาะตอนที่รถเคลื่อนที่
        dataset.append([round(dist_left, 2), round(dist_front, 2), round(dist_right, 2), action])

    pygame.display.flip()
    clock.tick(60) # รันที่ 60 FPS

# 9. บันทึก Dataset เป็นไฟล์ CSV เมื่อปิดโปรแกรม
pygame.quit()
print(f"Collected {len(dataset)-1} samples.")

with open('rover_navigation_dataset.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(dataset)
    
print("Dataset saved to 'rover_navigation_dataset.csv'")
sys.exit()