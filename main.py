import cv2
import mediapipe as mp
import numpy as np
import random
import time
from collections import Counter
import pygame

# Initialize pygame for sound effects (optional)
pygame.mixer.init()

class RockPaperScissorsCV:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils

        # Game variables
        self.player_score = 0
        self.ai_score = 0
        self.player_choice = "none"
        self.ai_choice = ""
        self.game_result = ""
        self.countdown = 0
        self.game_state = "choose_gesture"  # choose_gesture, countdown, show_result
        self.last_game_time = 0
        self.gesture_history = []

        # Hand tracking - simplified
        self.gesture_stable_count = 0
        self.stability_threshold = 15  # increased for better stability

        # Colors (BGR format for OpenCV)
        self.colors = {
            'bg': (20, 20, 40),
            'primary': (255, 100, 50),
            'secondary': (50, 255, 150),
            'accent': (255, 200, 0),
            'white': (255, 255, 255),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'purple': (255, 0, 255)
        }

    def detect_gesture(self, landmarks):
        """Simplified and more reliable gesture detection"""
        if not landmarks:
            return "none"

        # Get key landmark positions
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]

        index_tip = landmarks[8]
        index_pip = landmarks[6]

        middle_tip = landmarks[12]
        middle_pip = landmarks[10]

        ring_tip = landmarks[16]
        ring_pip = landmarks[14]

        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]

        # Simple finger up/down detection
        fingers = []

        # Thumb (different logic due to orientation)
        if thumb_tip.x > thumb_ip.x:  # Right hand
            fingers.append(1 if thumb_tip.x > thumb_ip.x else 0)
        else:  # Left hand
            fingers.append(1 if thumb_tip.x < thumb_ip.x else 0)

        # Other fingers - tip above pip joint
        fingers.append(1 if index_tip.y < index_pip.y else 0)
        fingers.append(1 if middle_tip.y < middle_pip.y else 0)
        fingers.append(1 if ring_tip.y < ring_pip.y else 0)
        fingers.append(1 if pinky_tip.y < pinky_pip.y else 0)

        # Determine gesture
        fingers_up = sum(fingers)

        if fingers_up == 0:
            return "rock"
        elif fingers_up >= 4:
            return "paper"
        elif fingers[1] == 1 and fingers[2] == 1 and sum(fingers[3:]) == 0:
            return "scissors"
        else:
            return "none"

    def get_ai_choice(self):
        """Generate AI choice with some strategy"""
        choices = ["rock", "paper", "scissors"]

        # Simple AI strategy: slightly favor counter to player's most common gesture
        if len(self.gesture_history) > 3:
            most_common = Counter(self.gesture_history[-5:]).most_common(1)[0][0]
            counters = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
            if random.random() < 0.4:  # 40% chance to counter
                return counters.get(most_common, random.choice(choices))

        return random.choice(choices)

    def determine_winner(self, player, ai):
        """Determine the winner of the round"""
        if player == ai:
            return "tie"
        elif (player == "rock" and ai == "scissors") or \
                (player == "paper" and ai == "rock") or \
                (player == "scissors" and ai == "paper"):
            return "player"
        else:
            return "ai"

    def draw_gradient_bg(self, img):
        """Draw an attractive gradient background"""
        h, w = img.shape[:2]
        gradient = np.zeros((h, w, 3), dtype=np.uint8)

        for i in range(h):
            ratio = i / h
            color = (
                int(self.colors['bg'][0] * (1 - ratio) + self.colors['primary'][0] * ratio * 0.3),
                int(self.colors['bg'][1] * (1 - ratio) + self.colors['primary'][1] * ratio * 0.3),
                int(self.colors['bg'][2] * (1 - ratio) + self.colors['primary'][2] * ratio * 0.3)
            )
            gradient[i, :] = color

        return cv2.addWeighted(img, 0.7, gradient, 0.3, 0)

    def draw_gesture_icon(self, img, gesture, pos, size=80, color=None):
        """Draw gesture icons"""
        x, y = pos
        if color is None:
            color = self.colors['white']

        if gesture == "rock":
            cv2.circle(img, (x, y), size//2, color, -1)
            cv2.circle(img, (x, y), size//2, self.colors['primary'], 3)
        elif gesture == "paper":
            cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), color, -1)
            cv2.rectangle(img, (x-size//2, y-size//2), (x+size//2, y+size//2), self.colors['primary'], 3)
        elif gesture == "scissors":
            cv2.line(img, (x-size//3, y-size//2), (x-size//3, y+size//2), color, 8)
            cv2.line(img, (x+size//3, y-size//2), (x+size//3, y+size//2), color, 8)
            cv2.circle(img, (x-size//3, y-size//2), 8, self.colors['primary'], -1)
            cv2.circle(img, (x+size//3, y-size//2), 8, self.colors['primary'], -1)

    def draw_ui(self, img):
        """Draw the game UI"""
        h, w = img.shape[:2]

        # Title
        cv2.putText(img, "ROCK PAPER SCISSORS AI", (w//2-200, 50),
                    cv2.FONT_HERSHEY_DUPLEX, 1.2, self.colors['accent'], 2)

        # Score
        score_text = f"YOU: {self.player_score}  |  AI: {self.ai_score}"
        cv2.putText(img, score_text, (w//2-120, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['white'], 2)

        # Game state specific UI
        if self.game_state == "choose_gesture":
            cv2.putText(img, "Make your gesture: Rock, Paper, or Scissors",
                        (w//2-220, h//2-60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['accent'], 2)

            if self.player_choice != "none":
                cv2.putText(img, f"Your Choice: {self.player_choice.upper()}",
                            (w//2-100, h//2-20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, self.colors['green'], 2)
                # Draw the chosen gesture icon
                self.draw_gesture_icon(img, self.player_choice, (w//2, h//2+30), 60, self.colors['green'])

                # Show stability progress
                progress = min(self.gesture_stable_count / self.stability_threshold, 1.0)
                bar_width = 200
                bar_x = w//2 - bar_width//2
                bar_y = h//2 + 100

                # Background bar
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (50, 50, 50), -1)
                # Progress bar
                cv2.rectangle(img, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 10), self.colors['green'], -1)

                cv2.putText(img, "Hold steady to confirm...",
                            (w//2-100, h//2+130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 1)
            else:
                cv2.putText(img, "Show: Rock (fist) | Paper (flat hand) | Scissors (two fingers)",
                            (w//2-270, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 1)

        elif self.game_state == "countdown":
            cv2.putText(img, f"Get Ready... {self.countdown}",
                        (w//2-120, h//2-40), cv2.FONT_HERSHEY_DUPLEX, 1.8, self.colors['accent'], 3)
            cv2.putText(img, f"Your Choice: {self.player_choice.upper()}",
                        (w//2-100, h//2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.colors['green'], 2)

        elif self.game_state == "show_result":
            # Show choices side by side
            cv2.putText(img, "YOU", (w//4-30, h//2-120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['white'], 2)
            cv2.putText(img, "AI", (3*w//4-20, h//2-120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.colors['white'], 2)

            # Draw gesture icons
            self.draw_gesture_icon(img, self.player_choice, (w//4, h//2-40), 80)
            self.draw_gesture_icon(img, self.ai_choice, (3*w//4, h//2-40), 80)

            # Show choice names
            cv2.putText(img, self.player_choice.upper(), (w//4-40, h//2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['white'], 2)
            cv2.putText(img, self.ai_choice.upper(), (3*w//4-40, h//2+30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['white'], 2)

            # Show result
            result_color = self.colors['green'] if self.game_result == "player" else \
                self.colors['red'] if self.game_result == "ai" else self.colors['accent']
            result_text = "YOU WIN!" if self.game_result == "player" else \
                "AI WINS!" if self.game_result == "ai" else "TIE!"

            cv2.putText(img, result_text, (w//2-80, h//2+100),
                        cv2.FONT_HERSHEY_DUPLEX, 1.4, result_color, 3)

            cv2.putText(img, "Press SPACE to play again",
                        (w//2-120, h//2+140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['white'], 1)

        # Instructions
        cv2.putText(img, "Controls: R=Reset Score | Q=Quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['secondary'], 1)

    def run(self):
        """Main game loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        print("🎮 Rock Paper Scissors CV Game Started!")
        print("📋 Instructions:")
        print("   1. Make Rock/Paper/Scissors gesture with your hand")
        print("   2. Hold the gesture steady to confirm")
        print("   3. Game will countdown and show results")
        print("   - Press R to reset scores")
        print("   - Press Q to quit")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]

            # Apply gradient background
            frame = self.draw_gradient_bg(frame)

            # Convert to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            # Reset detection variables
            current_gesture = "none"

            # Process detected hands
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # Draw hand landmarks
                    self.mp_drawing.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=self.colors['secondary'], thickness=2),
                        self.mp_drawing.DrawingSpec(color=self.colors['primary'], thickness=2)
                    )

                    # Check for gestures
                    gesture = self.detect_gesture(hand_landmarks.landmark)

                    if gesture in ["rock", "paper", "scissors"]:
                        current_gesture = gesture

                        # Show detected gesture
                        cv2.putText(frame, f"Detected: {gesture.upper()}", (50, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colors['secondary'], 2)

            # Handle game state transitions
            if self.game_state == "choose_gesture":
                if current_gesture in ["rock", "paper", "scissors"]:
                    if current_gesture == self.player_choice:
                        self.gesture_stable_count += 1
                        if self.gesture_stable_count >= self.stability_threshold:
                            print(f"✅ Player chose: {self.player_choice}")
                            self.game_state = "countdown"
                            self.countdown = 3
                            self.last_game_time = time.time()
                    else:
                        self.player_choice = current_gesture
                        self.gesture_stable_count = 0
                else:
                    self.gesture_stable_count = 0
                    if current_gesture == "none":
                        self.player_choice = "none"

            elif self.game_state == "countdown":
                current_time = time.time()
                if current_time - self.last_game_time >= 1:
                    self.countdown -= 1
                    self.last_game_time = current_time

                    if self.countdown <= 0:
                        # Generate AI choice and determine winner
                        self.ai_choice = self.get_ai_choice()
                        self.game_result = self.determine_winner(self.player_choice, self.ai_choice)

                        # Update scores
                        if self.game_result == "player":
                            self.player_score += 1
                        elif self.game_result == "ai":
                            self.ai_score += 1

                        # Add to history
                        self.gesture_history.append(self.player_choice)
                        if len(self.gesture_history) > 10:
                            self.gesture_history.pop(0)

                        self.game_state = "show_result"
                        print(f"🎯 Player: {self.player_choice} vs AI: {self.ai_choice} -> {self.game_result}")

            # Draw UI
            self.draw_ui(frame)

            # Show frame
            cv2.imshow('Rock Paper Scissors AI', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Spacebar - next round
                if self.game_state == "show_result":
                    self.reset_round()
            elif key == ord('r'):  # Reset scores
                self.player_score = 0
                self.ai_score = 0
                self.reset_round()
                print("🔄 Scores reset!")

        cap.release()
        cv2.destroyAllWindows()

    def reset_round(self):
        """Reset variables for next round"""
        self.game_state = "choose_gesture"
        self.player_choice = "none"
        self.ai_choice = ""
        self.game_result = ""
        self.gesture_stable_count = 0

if __name__ == "__main__":
    # Check if required packages are installed
    try:
        import cv2
        import mediapipe as mp
        import pygame
    except ImportError as e:
        print("❌ Missing required packages!")
        print("📦 Please install with: pip install opencv-python mediapipe pygame numpy")
        exit(1)

    print("🚀 Starting Rock Paper Scissors AI Game...")
    game = RockPaperScissorsCV()
    game.run()