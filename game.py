import pygame
import sys
import random
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
import threading
import numpy as np

# ─── CONFIG ───────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 480, 640
FPS = 60
GRAVITY = 0.4
PIPE_GAP = 170
PIPE_WIDTH = 70
PIPE_SPEED = 3
PIPE_SPAWN_RATE = 90  # frames
BIRD_X = 80

# Colors
SKY      = (113, 197, 207)
GROUND_C = (222, 216, 149)
PIPE_C   = (83, 170, 73)
PIPE_BD  = (57, 130, 50)
BIRD_C   = (255, 215, 0)
BIRD_BD  = (200, 150, 0)
EYE_C    = (255, 255, 255)
PUPIL_C  = (30,  30,  30)
WHITE    = (255, 255, 255)
BLACK    = (0,   0,   0)
RED      = (220, 50,  50)
DARK_SKY = (20,  30,  60)
SCORE_C  = (255, 255, 255)

# ─── HAND TRACKER (runs in background thread) ─────────────────────────────────
class HandTracker:
    def __init__(self):
        self.y_norm = 0.5          # 0.0 = top, 1.0 = bottom of camera frame
        self.hand_detected = False
        self.running = True
        self._lock = threading.Lock()

        self.cap = cv2.VideoCapture(0)
        
        # Try to use hand landmark detector, fallback to simple motion detection if unavailable
        try:
            base_options = python.BaseOptions(model_asset_path=None)  # Use default model
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                num_hands=1,
                min_hand_detection_confidence=0.7,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.detector = vision.HandLandmarker.create_from_options(options)
            self.use_hand_detector = True
        except Exception as e:
            print(f"Hand detector unavailable: {e}. Using fallback motion detection.")
            self.use_hand_detector = False
        
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        prev_gray = None
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            
            if self.use_hand_detector:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    from mediapipe import Image
                    img = Image(image_format=python.ImageFormat.SRGB, data=rgb)
                    detection_result = self.detector.detect(img)

                    if detection_result.hand_landmarks and len(detection_result.hand_landmarks) > 0:
                        landmarks = detection_result.hand_landmarks[0]
                        wrist = landmarks[0]
                        with self._lock:
                            self.y_norm = wrist.y
                            self.hand_detected = True
                    else:
                        with self._lock:
                            self.hand_detected = False
                except Exception as e:
                    with self._lock:
                        self.hand_detected = False
            else:
                # Fallback: simple skin color detection
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                lower_skin = np.array([0, 20, 70], dtype=np.uint8)
                upper_skin = np.array([20, 255, 255], dtype=np.uint8)
                skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
                
                contours, _ = cv2.findContours(skin_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    M = cv2.moments(largest)
                    if M['m00'] > 0:
                        cy = int(M['m01'] / M['m00'])
                        with self._lock:
                            self.y_norm = cy / h
                            self.hand_detected = True
                    else:
                        with self._lock:
                            self.hand_detected = False
                else:
                    with self._lock:
                        self.hand_detected = False

            cv2.putText(frame, "Gesture Flappy Bird", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)
            cv2.imshow("Hand Tracker (press Q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break

    def get_y(self):
        with self._lock:
            return self.y_norm, self.hand_detected

    def stop(self):
        self.running = False
        self.cap.release()
        cv2.destroyAllWindows()


# ─── GAME OBJECTS ─────────────────────────────────────────────────────────────
class Bird:
    RADIUS = 18

    def __init__(self):
        self.x = BIRD_X
        self.y = SCREEN_H // 2
        self.vel = 0
        self.angle = 0

    def update_gesture(self, y_norm):
        """Direct mapping: hand Y → bird Y (inverted: hand up = bird up)."""
        target_y = int(y_norm * (SCREEN_H - 100))   # map to playfield
        target_y = max(self.RADIUS, min(SCREEN_H - 100, target_y))
        # Smooth follow
        diff = target_y - self.y
        self.vel = diff * 0.18
        self.y += self.vel
        self.angle = max(-30, min(30, self.vel * 3))

    def draw(self, surf):
        cx, cy = int(self.x), int(self.y)
        r = self.RADIUS
        # Body
        pygame.draw.circle(surf, BIRD_BD, (cx, cy), r + 2)
        pygame.draw.circle(surf, BIRD_C,  (cx, cy), r)
        # Wing
        pygame.draw.ellipse(surf, (255, 180, 0),
                            (cx - r + 2, cy + 4, r + 4, 10))
        # Eye
        pygame.draw.circle(surf, EYE_C,  (cx + 8, cy - 5), 6)
        pygame.draw.circle(surf, PUPIL_C,(cx + 10, cy - 5), 3)
        # Beak
        pygame.draw.polygon(surf, (255, 140, 0), [
            (cx + r - 4, cy),
            (cx + r + 10, cy + 3),
            (cx + r - 4, cy + 7)
        ])

    def get_rect(self):
        r = self.RADIUS - 4   # slightly smaller hitbox
        return pygame.Rect(self.x - r, self.y - r, r * 2, r * 2)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.gap_y = random.randint(160, SCREEN_H - 160 - PIPE_GAP)
        self.scored = False

    def update(self):
        self.x -= PIPE_SPEED

    def draw(self, surf):
        top_h = self.gap_y
        bot_y  = self.gap_y + PIPE_GAP
        bot_h  = SCREEN_H - 100 - bot_y

        # Top pipe
        pygame.draw.rect(surf, PIPE_BD, (self.x - 2, 0, PIPE_WIDTH + 4, top_h + 4))
        pygame.draw.rect(surf, PIPE_C,  (self.x,     0, PIPE_WIDTH,     top_h))
        # Top cap
        pygame.draw.rect(surf, PIPE_BD, (self.x - 6, top_h - 24, PIPE_WIDTH + 12, 28))
        pygame.draw.rect(surf, PIPE_C,  (self.x - 4, top_h - 22, PIPE_WIDTH + 8,  24))

        # Bottom pipe
        pygame.draw.rect(surf, PIPE_BD, (self.x - 2, bot_y - 4, PIPE_WIDTH + 4, bot_h + 6))
        pygame.draw.rect(surf, PIPE_C,  (self.x,     bot_y,     PIPE_WIDTH,     bot_h))
        # Bottom cap
        pygame.draw.rect(surf, PIPE_BD, (self.x - 6, bot_y - 4, PIPE_WIDTH + 12, 28))
        pygame.draw.rect(surf, PIPE_C,  (self.x - 4, bot_y - 2, PIPE_WIDTH + 8,  24))

    def collides(self, bird_rect):
        top_rect = pygame.Rect(self.x, 0, PIPE_WIDTH, self.gap_y)
        bot_rect = pygame.Rect(self.x, self.gap_y + PIPE_GAP,
                               PIPE_WIDTH, SCREEN_H)
        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bot_rect)

    def off_screen(self):
        return self.x + PIPE_WIDTH < 0


# ─── CLOUDS ───────────────────────────────────────────────────────────────────
class Cloud:
    def __init__(self, x=None):
        self.x = x if x else random.randint(0, SCREEN_W)
        self.y = random.randint(40, 200)
        self.speed = random.uniform(0.3, 0.8)
        self.scale = random.uniform(0.6, 1.2)

    def update(self):
        self.x -= self.speed
        if self.x < -120:
            self.x = SCREEN_W + 20
            self.y = random.randint(40, 200)

    def draw(self, surf):
        cx, cy = int(self.x), int(self.y)
        s = self.scale
        color = (255, 255, 255)
        for ox, oy, r in [(0,0,22),(25,-10,30),(50,0,22),(-15,8,18),(65,8,18)]:
            pygame.draw.circle(surf, color,
                               (cx + int(ox*s), cy + int(oy*s)), int(r*s))


# ─── MAIN GAME ────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("🐦 Gesture Flappy Bird")
    clock = pygame.time.Clock()

    font_big   = pygame.font.SysFont("Arial", 52, bold=True)
    font_med   = pygame.font.SysFont("Arial", 28, bold=True)
    font_small = pygame.font.SysFont("Arial", 20)

    tracker = HandTracker()

    # Ground tiles
    GROUND_H = 100
    ground_scroll = 0

    def make_stars():
        return [(random.randint(0, SCREEN_W), random.randint(0, SCREEN_H-GROUND_H),
                 random.randint(1, 3)) for _ in range(60)]
    stars = make_stars()

    def reset():
        return (
            Bird(),
            [Cloud(x=random.randint(0, SCREEN_W)) for _ in range(5)],
            [],   # pipes
            0,    # score
            0,    # frame count
            "waiting",  # state: waiting | playing | dead
        )

    bird, clouds, pipes, score, frame_count, state = reset()
    high_score = 0

    while True:
        dt = clock.tick(FPS)
        y_norm, hand_ok = tracker.get_y()

        # ── Events ──
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                tracker.stop(); pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    tracker.stop(); pygame.quit(); sys.exit()
                if ev.key == pygame.K_r and state == "dead":
                    bird, clouds, pipes, score, frame_count, state = reset()
                if ev.key == pygame.K_SPACE and state == "waiting":
                    state = "playing"
            if ev.type == pygame.MOUSEBUTTONDOWN:
                if state == "waiting":
                    state = "playing"
                elif state == "dead":
                    bird, clouds, pipes, score, frame_count, state = reset()

        # Start game when hand detected
        if state == "waiting" and hand_ok:
            state = "playing"

        # ── Update ──
        for c in clouds:
            c.update()

        if state == "playing":
            frame_count += 1
            bird.update_gesture(y_norm)

            # Spawn pipes
            if frame_count % PIPE_SPAWN_RATE == 0:
                pipes.append(Pipe(SCREEN_W + 10))

            for p in pipes:
                p.update()
                if not p.scored and p.x + PIPE_WIDTH < bird.x:
                    score += 1
                    p.scored = True
                if p.collides(bird.get_rect()):
                    state = "dead"
                    high_score = max(high_score, score)

            pipes = [p for p in pipes if not p.off_screen()]

            # Ground collision
            if bird.y + bird.RADIUS >= SCREEN_H - GROUND_H:
                state = "dead"
                high_score = max(high_score, score)

            # Ceiling
            bird.y = max(bird.RADIUS, bird.y)

            ground_scroll = (ground_scroll - PIPE_SPEED) % 40

        # ── Draw ──
        # Sky gradient
        for row in range(SCREEN_H - GROUND_H):
            t = row / (SCREEN_H - GROUND_H)
            r = int(113 + t * 30)
            g = int(197 + t * 10)
            b = int(207 - t * 20)
            pygame.draw.line(screen, (r, g, b), (0, row), (SCREEN_W, row))

        for cx, cy, cr in stars:
            pygame.draw.circle(screen, (255,255,255,80), (cx, cy), cr)

        for c in clouds:
            c.draw(screen)

        for p in pipes:
            p.draw(screen)

        # Ground
        ground_y = SCREEN_H - GROUND_H
        pygame.draw.rect(screen, (150, 130, 80), (0, ground_y, SCREEN_W, GROUND_H))
        pygame.draw.rect(screen, GROUND_C,       (0, ground_y + 10, SCREEN_W, GROUND_H))
        # Ground stripes
        for i in range(-1, SCREEN_W // 40 + 2):
            gx = i * 40 + int(ground_scroll)
            pygame.draw.rect(screen, (200, 195, 130), (gx, ground_y + 10, 20, 8))

        bird.draw(screen)

        # ── HUD ──
        # Score
        score_surf = font_big.render(str(score), True, WHITE)
        score_sh   = font_big.render(str(score), True, (0,0,0))
        screen.blit(score_sh, (SCREEN_W//2 - score_surf.get_width()//2 + 2, 32))
        screen.blit(score_surf, (SCREEN_W//2 - score_surf.get_width()//2, 30))

        # Hand status indicator
        dot_c = (80, 255, 80) if hand_ok else (255, 80, 80)
        pygame.draw.circle(screen, dot_c, (18, 18), 8)
        label = font_small.render("Hand OK" if hand_ok else "No Hand", True, dot_c)
        screen.blit(label, (30, 10))

        # ── SCREENS ──
        if state == "waiting":
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 120))
            screen.blit(overlay, (0, 0))

            title = font_big.render("FLAPPY BIRD", True, (255, 215, 0))
            screen.blit(title, (SCREEN_W//2 - title.get_width()//2, 160))

            sub = font_med.render("Show your hand to start!", True, WHITE)
            screen.blit(sub, (SCREEN_W//2 - sub.get_width()//2, 240))

            tip1 = font_small.render("✋ Raise hand  →  Bird goes UP", True, (200, 255, 200))
            tip2 = font_small.render("✋ Lower hand  →  Bird goes DOWN", True, (200, 200, 255))
            screen.blit(tip1, (SCREEN_W//2 - tip1.get_width()//2, 300))
            screen.blit(tip2, (SCREEN_W//2 - tip2.get_width()//2, 330))

            click = font_small.render("(or click / press SPACE)", True, (180, 180, 180))
            screen.blit(click, (SCREEN_W//2 - click.get_width()//2, 380))

        elif state == "dead":
            overlay = pygame.Surface((SCREEN_W, SCREEN_H), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 150))
            screen.blit(overlay, (0, 0))

            go = font_big.render("GAME OVER", True, RED)
            screen.blit(go, (SCREEN_W//2 - go.get_width()//2, 180))

            sc  = font_med.render(f"Score:  {score}", True, WHITE)
            hsc = font_med.render(f"Best:   {high_score}", True, (255, 215, 0))
            screen.blit(sc,  (SCREEN_W//2 - sc.get_width()//2,  260))
            screen.blit(hsc, (SCREEN_W//2 - hsc.get_width()//2, 300))

            rst = font_small.render("Click or press R to restart", True, (180, 255, 180))
            screen.blit(rst, (SCREEN_W//2 - rst.get_width()//2, 360))

        pygame.display.flip()

    tracker.stop()


if __name__ == "__main__":
    main()
