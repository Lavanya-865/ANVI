import pyttsx3
import threading
import queue
import time

class VoiceAssistant:
    """
    Thread-safe, non-blocking text-to-speech helper.
    Prevents the camera loop from freezing and stops pyttsx3 from crashing.
    """
    def __init__(self, rate=200, repeat_interval=1.5, voice_index=2, volume=1.0):
        self.rate = rate
        self.volume = volume
        self.repeat_interval = repeat_interval
        self.voice_index = voice_index
        
        self._queue = queue.Queue()
        # Start a background thread to handle the speech without blocking the camera
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        
        print("[VOICE] Threaded voice assistant initialized.")

    def _worker(self):
        last_text = None
        last_time = 0.0
        
        while True:
            text = self._queue.get() 
            if text is None: 
                break
                
            now = time.time()
            changed = text != last_text
            enough_time_passed = (now - last_time) >= self.repeat_interval
            
            # Prevent the engine from being spammed 30 times a second
            if not (changed or enough_time_passed):
                continue
                
            last_text = text
            last_time = now
            print(f"[ANVI VOICE] {text}")
            
            # Create a fresh engine inside the thread for maximum reliability
            engine = pyttsx3.init()
            engine.setProperty('rate', self.rate)
            engine.setProperty('volume', self.volume)
            
            # Apply your requested voice index
            voices = engine.getProperty('voices')
            if voices:
                # Use the requested index, or fallback to 0 if the index doesn't exist on this PC
                idx = self.voice_index if len(voices) > self.voice_index else 0 
                engine.setProperty('voice', voices[idx].id)
            
            try:
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[VOICE ERROR] {e}")
            finally:
                engine.stop()
                del engine

    def speak(self, text):
        if not text:
            return
            
        # Clear out stale instructions so the voice never lags behind your hand
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
                
        self._queue.put(text)

    def stop(self):
        self._queue.put(None)

if __name__ == "__main__":
    print("================================")
    print("      ANVI VOICE TEST")
    print("================================")
    
    voice = VoiceAssistant(rate=200, voice_index=2)
    voice.speak("Hello. This is ANVI. Voice is working.")
    time.sleep(2)
    voice.speak("Please move your hand to the right.")
    time.sleep(2)
    voice.stop()
    print("Voice test completed.")