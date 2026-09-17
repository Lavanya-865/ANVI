import cv2
import numpy as np
from google import genai
from google.genai import types
import json
import math
import time
import speech_recognition as sr

# Import your existing modules
from hand_tracking import HandTracker
from voice_assistant import VoiceAssistant

# ==========================================
# CONFIGURATION
# ==========================================
API_KEY = "AIzaSyAb7dtKvDFnib8T36ral-0PLT-7iMXfslQ"
MODEL_ID = "gemini-3.6-flash"

client = genai.Client(api_key=API_KEY)

def get_appliance_steps(image_bytes, user_goal):
    prompt_text = f"""
    You are an Assistive Navigation Agent for visually impaired users.
    User Goal: "{user_goal}"
    Target Language: "English"
    Examine the appliance control panel image:
    1. Identify the appliance type.
    2. Determine the EXACT sequential order of buttons/controls that MUST be pressed to achieve the user's goal.
    
    CRITICAL RULES: 
    * You can group multiple presses of the SAME button into a single step (e.g., "Press the 10s button 3 times").
    * box_2d scale: 0-1000 (bounding box of the specific control).
    
    Return PURE JSON in this exact structure:
    {{
        "device_name": "Appliance Name",
        "steps": [
            {{
                "order": 1,
                "button_name": "Timer Button",
                "text": "Locate and press the 30 sec timer button 3 times",
                "action_type": "tap", 
                "box_2d": [ymin, xmin, ymax, xmax] 
            }}
        ]
    }}
    """
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt_text
            ],
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )
        data = json.loads(response.text)
        return data[0] if isinstance(data, list) else data
    except Exception as e:
        print(f"API Error: {e}")
        return None

# ==========================================
# MICROPHONE SETUP
# ==========================================
recognizer = sr.Recognizer()

# Increased to 2.0 so the system waits longer before cutting you off
recognizer.pause_threshold = 2.0 
recognizer.dynamic_energy_threshold = True

mic = sr.Microphone()
user_said_done = False

def mic_callback(recognizer, audio):
    """This runs in a background thread whenever sound is detected."""
    global user_said_done
    try:
        text = recognizer.recognize_google(audio).lower()
        print(f"[Mic heard]: {text}")
        
        keywords = ["done", "yes", "pressed", "next", "ready", "ok", "okay", "yep"]
        if any(word in text for word in keywords):
            user_said_done = True
    except sr.UnknownValueError:
        pass
    except sr.RequestError:
        pass

print("Calibrating microphone for ambient noise... Please wait.")
with mic as source:
    recognizer.adjust_for_ambient_noise(source, duration=2.0)
print("Microphone ready.")

# ==========================================
# INITIALIZATION
# ==========================================
tracker = HandTracker()
# Increased speed to 220 as requested previously
voice = VoiceAssistant(
    rate=200,
    repeat_interval=1.5,
    voice_index=2,   # Microsoft Zira - Female US
    volume=1.0
)
orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
bf = cv2.BFMatcher(cv2.NORM_HAMMING)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

reference_img = None
reference_gray = None
kp_ref, des_ref = None, None
steps_data = []
current_step = 0
threshold = 30

audio_pause_until = 0.0 
awaiting_confirmation = False
last_prompt_time = 0.0
stop_listening = None # Will hold our background mic thread later

# ==========================================
# PHASE 1: CAPTURE & AUDIO GOAL PROCESSING
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break
    
    cv2.putText(frame, "Align panel and press 'c' to capture", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Setup Phase", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('c'):
        reference_img = frame.copy()
        reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
        kp_ref, des_ref = orb.detectAndCompute(reference_gray, None)
        
        _, encoded_img = cv2.imencode('.jpg', reference_img)
        raw_bytes = encoded_img.tobytes()
        
        # --- NEW VERBAL GOAL LOGIC ---
        cv2.putText(frame, "Listening for your goal...", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("Setup Phase", frame)
        cv2.waitKey(1) # Force UI to update
        
        voice.speak("Panel captured. What would you like to do?")
        time.sleep(3.5) # Wait for the TTS to finish speaking
        
        goal = ""
        while not goal:
            try:
                print("Listening for goal...")
                with mic as source:
                    audio = recognizer.listen(source, timeout=8, phrase_time_limit=15)
                goal = recognizer.recognize_google(audio)
                print(f"Goal heard: {goal}")
                
                voice.speak(f"Understood. Getting instructions to {goal}.")
                time.sleep(4)
                
            except sr.WaitTimeoutError:
                voice.speak("I didn't hear anything. What would you like to do?")
                time.sleep(3.5)
            except sr.UnknownValueError:
                voice.speak("I didn't catch that. Please say it again.")
                time.sleep(3)
            except sr.RequestError:
                print("Network error connecting to speech recognition.")
                break
        
        if not goal:
            break
            
        print("Sending to Gemini API...")
        
        api_data = get_appliance_steps(raw_bytes, goal)
        if api_data and "steps" in api_data:
            steps = api_data["steps"]
            img_h, img_w = reference_img.shape[:2]
            
            for step in steps:
                ymin, xmin, ymax, xmax = step["box_2d"]
                center_x = int(((xmin + xmax) / 2000.0) * img_w)
                center_y = int(((ymin + ymax) / 2000.0) * img_h)
                steps_data.append({
                    "name": step["button_name"],
                    "instruction": step["text"],
                    "ref_coord": (center_x, center_y)
                })
            
            print(f"Successfully mapped {len(steps_data)} steps.")
            cv2.destroyWindow("Setup Phase")
            
            # Start the background listener only AFTER the initial goal is captured
            stop_listening = recognizer.listen_in_background(mic, mic_callback)
            break
        else:
            print("Failed to get valid steps. Try capturing again.")

# ==========================================
# PHASE 2: LIVE NAVIGATION LOOP
# ==========================================
print("Starting Live Navigation...")
if steps_data:
    first_msg = f"Starting step 1. {steps_data[0]['instruction']}"
    voice.speak(first_msg)
    audio_pause_until = time.time() + 4.0

while True:
    ret, frame = cap.read()
    if not ret: break
    
    if current_step >= len(steps_data):
        cv2.putText(frame, "Operation Complete", (120, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        cv2.imshow("Navigation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue

    current_target_data = steps_data[current_step]
    ref_target_pt = np.float32([[current_target_data["ref_coord"]]]).reshape(-1, 1, 2)
    
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_frame, des_frame = orb.detectAndCompute(gray_frame, None)
    
    live_target_coord = None
    if des_frame is not None and des_ref is not None:
        matches = bf.knnMatch(des_ref, des_frame, k=2)
        good_matches = [m for match in matches if len(match)==2 for m, n in [match] if m.distance < 0.70 * n.distance]
        
        if len(good_matches) >= 30:
            src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                live_pt = cv2.perspectiveTransform(ref_target_pt, H)
                live_target_coord = (int(live_pt[0][0][0]), int(live_pt[0][0][1]))

    results = tracker.process(frame)
    instruction = ""
    is_audio_paused = time.time() < audio_pause_until
    
    if live_target_coord:
        target_x, target_y = live_target_coord
        cv2.circle(frame, live_target_coord, 18, (0, 0, 255), -1)
        cv2.putText(frame, current_target_data["name"], (target_x - 25, target_y - 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                tracker.draw_landmarks(frame, hand_landmarks)
                finger_x, finger_y = tracker.get_fingertip(frame, hand_landmarks)
                cv2.circle(frame, (finger_x, finger_y), 12, (0, 255, 0), -1)
                
                dx = target_x - finger_x
                dy = target_y - finger_y
                distance = math.sqrt(dx*dx + dy*dy)
                
                if awaiting_confirmation:
                    cv2.putText(frame, "Waiting for you to finish the step...", (30, 180), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                    
                    if user_said_done:
                        current_step += 1
                        awaiting_confirmation = False
                        user_said_done = False
                        
                        if current_step < len(steps_data):
                            next_inst = steps_data[current_step]["instruction"]
                            transition_msg = f"Step complete. Starting next step. {next_inst}"
                            voice.speak(transition_msg)
                            audio_pause_until = time.time() + 5.0
                        else:
                            voice.speak("All steps completed successfully.")
                            audio_pause_until = time.time() + 5.0
                            
                    elif time.time() - last_prompt_time > 20.0:
                        voice.speak("Please let me know when you have finished the step.")
                        last_prompt_time = time.time()
                
                else:
                    if distance < threshold:
                        if not is_audio_paused:
                            voice.speak("Button reached. Please complete the step and let me know.")
                            awaiting_confirmation = True
                            user_said_done = False
                            last_prompt_time = time.time()
                            audio_pause_until = time.time() + 3.0
                    else:
                        if not is_audio_paused:
                            if abs(dx) > abs(dy):
                                instruction = "Move Right" if dx > threshold else "Move Left"
                            else:
                                instruction = "Move Down" if dy > threshold else "Move Up"
                        
                        if instruction:
                            voice.speak(instruction)
                            cv2.line(frame, (finger_x, finger_y), live_target_coord, (255, 255, 0), 2)
                            cv2.putText(frame, instruction, (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    else:
        cv2.putText(frame, "Searching for panel...", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    status_text = "Listening to instruction..." if is_audio_paused else "Tracking..."
    if awaiting_confirmation: status_text = "Awaiting verbal confirmation..."
    
    cv2.putText(frame, status_text, (30, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(frame, f"Step: {current_step + 1}/{len(steps_data)}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Navigation", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if stop_listening:
    stop_listening(wait_for_stop=False)
voice.stop()
cap.release()
cv2.destroyAllWindows