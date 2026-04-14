import pygame
import math

pygame.init()
screen = pygame.display.set_mode((800, 800))
clock = pygame.time.Clock()

center_x, center_y = screen.get_width()//2, screen.get_height()//2  
hex_radius = min(center_x, center_y) *0.8 - 50 
wall_thickness = 10

ball_rad = 20
x, y = center_x + hex_radius //2, center_y
vx, vy = 8, -12 # initial velocity (needs to be enough speed)
gravity = 0.1
friction = 0.95

rotation_angle_degrees = 0

running = True  
while running: 
    for event in pygame.event.get():
        if event.type == pygame.QUIT: 
            running = False
            
    # Update velocity and position    
    vy += gravity  
    x += vx
    y += vy
    
    # Check collision with each wall segment
    collision = False
    for i in range(6):        
        start_a = rotation_angle_degrees + i*60 %360
        end_a = start_a + 60
        
        a_x = center_x + hex_radius * math.cos(math.radians(start_a))
        a_y = center_y + hex_radius * math.sin(math.radians(start_a))                
        
        b_x = center_x + hex_radius * math.cos(math.radians(end_a))
        b_y = center_y + hex_radius * math.sin(math.radians(end_a))
        
        mid_m_x = (a_x + b_x)/2
        mid_m_y = (a_y + b_y)/2
        
        # Compute CM vector and normalize for normal direction N
        cmx = mid_m_x - center_x 
        cmy = mid_m_y - center_y
        
        cm_len = math.hypot(cmx, cmy)        
        n_x = cmx/cm_len
        n_y = cmy/cm_len # points from midpoint towards center (inward normal)
        
        proj = ((x - mid_m_x)*n_x + (y - mid_m_y)*n_y)
        
        if proj > ball_rad:  
            collision_dist = proj - ball_rad
            x -= n_x * collision_dist 
            y -= n_y * collision_dist
            
            dot_p = vx*n_x + vy*n_y
            refl_n = -2*dot_p*n_x
            refl_tang = vy*n_x - vx*n_y # tangent component
            
            vx = (refl_n)*friction - (refl_tang)*n_y 
            vy = (refl_n)*friction + (refl_tang)*n_x  
            
            collision = True
            break
    
    screen.fill((0,0,0))
    
    # Draw hexagon walls at current rotation_angle
    for i in range(6):        
        start_a = rotation_angle_degrees + i*60 %360 
        end_a = start_a + 60
        
        a_x = center_x + hex_radius * math.cos(math.radians(start_a))
        a_y = center_y + hex_radius * math.sin(math.radians(start_a))                
        
        b_x = center_x + hex_radius * math.cos(math.radians(end_a))
        b_y = center_y + hex_radius * math.sin(math.radians(end_a))               
        
        pygame.draw.line(screen, (255,0,0), (a_x, a_y), (b_x, b_y))
    
    # Draw the bouncing ball
    pygame.draw.circle(screen, (0,0,255), (int(x), int(y)), ball_rad)
    
    rotation_angle_degrees = (rotation_angle_degrees + 2) %360
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit() 
