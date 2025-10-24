import cv2
import google.generativeai as genai
import time
import pyttsx3
import os
import sys
import base64
from pathlib import Path

# ==============================
# WINDOWS CONSOLE ENCODING
# ==============================
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ==============================
# LOAD ENVIRONMENT VARIABLES
# ==============================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("[INFO] python-dotenv not installed. Install with: pip install python-dotenv")

# ==============================
# CONFIGURATION
# ==============================
API_KEY = (
    os.getenv("GEMINI_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
if not API_KEY:
    raise ValueError(
        "❌ GEMINI_API_KEY not found. Set GEMINI_API_KEY in your environment or .env file.\n"
        "Example: GEMINI_API_KEY=your_key_here\n"
    )
 
MODEL_NAME = "gemini-2.5-flash"
genai.configure(api_key=API_KEY)

# ==============================
# TEXT-TO-SPEECH SETUP
# ==============================
class Speaker:
    def __init__(self, rate=170):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', rate)
            self.enabled = True
        except Exception as e:
            print(f"⚠️ TTS initialization failed: {e}")
            self.enabled = False

    def speak(self, text, print_text=True):
        if print_text:
            print(f"AI: {text}")
        if self.enabled:
            try:
                self.engine.say(text)
                self.engine.runAndWait()
            except Exception as e:
                print(f"TTS Error: {e}")

# ==============================
# IMAGE ANALYSIS
# ==============================
class ObstacleDetector:
    def __init__(self, model_name=MODEL_NAME):
        self.model = genai.GenerativeModel(model_name)
        self.prompt = (
            "You are assisting a visually impaired person. "
            "Analyze this image and describe any obstacles directly in the person's path. "
            "If there are obstacles, specify their position (left, center, or right). "
            "Be concise and actionable. Examples: 'Clear path ahead', "
            "'Obstacle on right, move left', 'Person directly ahead, stop'."
        )

    def analyze_frame(self, frame):
        """Encode frame and send to Gemini for analysis."""
        success, encoded_img = cv2.imencode('.jpg', frame)
        if not success:
            raise ValueError("Failed to encode frame")

        image_bytes = encoded_img.tobytes()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        response = self.model.generate_content([
            self.prompt,
            {"inline_data": {"mime_type": "image/jpeg", "data": image_base64}}
        ])

        if hasattr(response, "text") and response.text:
            return response.text.strip()
        else:
            return "No clear response from AI."

# ==============================
# MAIN APPLICATION
# ==============================
class AssistiveAI:
    def __init__(self, camera_id=0, interval=3):
        self.camera_id = camera_id
        self.interval = interval
        self.speaker = Speaker()
        self.detector = ObstacleDetector()
        self.cap = None

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print("[ERROR] Could not open webcam.")
            return

        print("=" * 50)
        print("ASSISTIVE AI SYSTEM - ACTIVE")
        print("=" * 50)
        print(f"Analyzing frames every {self.interval} seconds")
        print("Press 'Q' to quit\n")

        self.speaker.speak("Assistive AI system activated.")

        last_analysis = 0
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[ERROR] Could not read frame.")
                    break

                current_time = time.time()
                if current_time - last_analysis >= self.interval:
                    last_analysis = current_time
                    self._analyze_and_speak(frame)

                cv2.imshow('Assistive AI - Live Feed (Press Q to quit)', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.speaker.speak("Shutting down assistive AI system.")
                    break

        except KeyboardInterrupt:
            print("\n[INTERRUPTED] User stopped the system")
            self.speaker.speak("System interrupted.")
        finally:
            self.cleanup()

    def _analyze_and_speak(self, frame):
        try:
            print("\n[ANALYZING] Processing frame...")
            result = self.detector.analyze_frame(frame)
            self.speaker.speak(result)
        except Exception as e:
            print(f"[WARNING] Analysis error: {e}")
            self.speaker.speak("Error analyzing the scene.")

    def cleanup(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 50)
        print("SYSTEM SHUTDOWN COMPLETE")
        print("=" * 50)

# ==============================
# ENTRY POINT
# ==============================
def main():
    app = AssistiveAI(camera_id=0, interval=3)
    app.start()

if __name__ == "__main__":
    main()
