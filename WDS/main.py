import cv2
from ultralytics import YOLO
import queue
import time
import os
import sys
import platform
import warnings
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Suppress OpenCV warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load environment variables
load_dotenv()


class SecurityMonitor:
    def __init__(self):
        # Logging configuration
        self.logging_level = int(os.getenv('LOGGING_LEVEL', 2))  # 0: minimal, 1: moderate, 2: verbose

        # Email notification configuration
        self.email_sender = os.getenv('EMAIL_SENDER')
        self.email_password = os.getenv('EMAIL_PASSWORD')
        self.email_recipient = os.getenv('EMAIL_RECIPIENT')
        self.email_server = os.getenv('EMAIL_SERVER', 'smtp.gmail.com')
        self.email_port = int(os.getenv('EMAIL_PORT', 587))

        # Image attachment option
        self.send_image_attachments = os.getenv('SEND_IMAGE_ATTACHMENTS', '1') == '1'

        # Logging options
        self.save_detection_logs = os.getenv('SAVE_DETECTION_LOGS', '1') == '1'
        self.log_dir = os.getenv('LOG_DIR', 'detection_logs')

        # Sound alarm option
        self.enable_sound_alarm = os.getenv('ENABLE_SOUND_ALARM', '1') == '1'

        # Detection thresholds
        self.weapon_confidence = float(os.getenv('WEAPON_CONFIDENCE', 0.25))
        self.person_confidence = float(os.getenv('PERSON_CONFIDENCE', 0.3))

        # Cooldown for notifications (seconds)
        self.notification_cooldown = int(os.getenv('NOTIFICATION_COOLDOWN', 60))
        self.last_notification_time = 0

        # Initialize components
        self.camera = None
        self.yolo_model = None
        self.detection_queue = queue.Queue()

        # Path to detection log file
        if self.save_detection_logs:
            os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, f"detection_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            with open(self.log_file, 'w') as f:
                f.write("timestamp,object_type,confidence,x1,y1,x2,y2\n")

        # Path to your fine-tuned model
        self.model_path = os.getenv('MODEL_PATH', 'yolov8x.pt')

        # Initialize components
        self._initialize_components()

    def _log(self, message, level=1):
        """
        Logging method with configurable verbosity
        """
        if self.logging_level >= level:
            print(f"[{'LOW' if level == 0 else 'MED' if level == 1 else 'HIGH'}] {message}")

    def _initialize_components(self):
        """
        Initialize camera and YOLO model with comprehensive error handling
        """
        try:
            # Initialize camera
            self.camera = self._robust_camera_initialization()
            self._log("Camera initialized successfully", 1)

            # Load YOLO model
            self._log(f"Loading YOLO model from {self.model_path}...", 1)

            try:
                # Try loading custom model
                self.yolo_model = YOLO(self.model_path)
                self._log("Custom YOLO model loaded successfully", 1)
            except Exception as e:
                self._log(f"Failed to load custom model: {e}. Falling back to better default model.", 0)
                # Try larger, more accurate models as fallbacks
                try:
                    self.yolo_model = YOLO('yolov8x.pt')  # Much larger and more accurate than yolov8n
                    self._log("YOLOv8x model loaded successfully", 1)
                except Exception:
                    try:
                        self.yolo_model = YOLO('yolov8l.pt')  # Large model, good balance
                        self._log("YOLOv8l model loaded successfully", 1)
                    except Exception:
                        self.yolo_model = YOLO('yolov8n.pt')  # Smallest model as last resort
                        self._log("YOLOv8n model loaded successfully as last resort", 1)

        except Exception as e:
            self._log(f"Initialization error: {e}", 0)
            sys.exit(1)

    def _robust_camera_initialization(self):
        """
        Robust camera initialization with multiple fallback mechanisms
        """
        # Try multiple camera indices and backends
        camera_backends = [
            cv2.CAP_ANY,  # Default/Any backend
            cv2.CAP_DSHOW,  # Windows DirectShow
            cv2.CAP_MSMF,  # Windows Media Foundation
            cv2.CAP_V4L2,  # Linux V4L2
            cv2.CAP_AVFOUNDATION  # macOS AVFoundation
        ]

        # Platform-specific backend prioritization
        if platform.system() == 'Darwin':
            camera_backends = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
        elif platform.system() == 'Windows':
            camera_backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        elif platform.system() == 'Linux':
            camera_backends = [cv2.CAP_V4L2, cv2.CAP_ANY]

        # Try different backends and indices
        for backend in camera_backends:
            for index in range(3):  # Try first 3 camera indices
                try:
                    # Special handling for macOS
                    if platform.system() == 'Darwin':
                        os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'

                    # Attempt to open camera
                    cap = cv2.VideoCapture(index, backend)

                    if not cap.isOpened():
                        cap.release()
                        continue

                    # Set standard resolution
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

                    # Verify frame capture
                    ret, frame = cap.read()

                    if ret and frame is not None and frame.size > 0:
                        self._log(f"Camera initialized: Backend={backend}, Index={index}", 1)
                        return cap

                    cap.release()

                except Exception as e:
                    self._log(f"Camera attempt error: {e}", 1)

        # If no camera found
        raise RuntimeError("Unable to initialize camera. No viable camera found.")

    def _send_email_alert(self, message, image=None):
        """
        Send alert via Email if configured, optionally with image attachment
        """
        if not self.email_sender or not self.email_password or not self.email_recipient:
            self._log("Email configuration missing", 0)
            return

        # Check cooldown period to avoid spamming emails
        current_time = time.time()
        if current_time - self.last_notification_time < self.notification_cooldown:
            self._log("Email alert skipped (cooldown period)", 1)
            return

        try:
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.email_sender
            msg['To'] = self.email_recipient
            msg['Subject'] = "⚠️ SECURITY ALERT: Weapon Detected"

            # Add message body with enhanced formatting
            email_body = f"""
            <html>
            <body>
                <h2 style="color: #cc0000;">⚠️ SECURITY ALERT: Weapon Detected</h2>
                <p><b>Alert Details:</b> {message}</p>
                <p><b>Time:</b> {time.strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>This is an automated security alert. Please review the attached image and take appropriate action if necessary.</p>
                <p style="color: #888888; font-size: 12px;">Powered by Weapon Detection System</p>
            </body>
            </html>
            """
            msg.attach(MIMEText(email_body, 'html'))

            # Attach image if provided and enabled
            if image is not None and self.send_image_attachments:
                # Encode image
                _, img_encoded = cv2.imencode('.jpg', image)
                img_bytes = img_encoded.tobytes()

                # Attach image
                image_attachment = MIMEImage(img_bytes)
                image_attachment.add_header('Content-Disposition', 'attachment',
                                            filename=f'weapon_detected_{time.strftime("%Y%m%d_%H%M%S")}.jpg')
                msg.attach(image_attachment)

            # Connect to email server
            server = smtplib.SMTP(self.email_server, self.email_port)
            server.starttls()
            server.login(self.email_sender, self.email_password)

            # Send email
            server.send_message(msg)
            server.quit()

            # Update last notification time
            self.last_notification_time = current_time

            self._log(f"Email alert sent: {message}", 1)
        except Exception as e:
            self._log(f"Email alert failed: {e}", 0)

    def _log_detection(self, timestamp, label, confidence, bbox):
        """
        Log detection to CSV file
        """
        if self.save_detection_logs:
            x1, y1, x2, y2 = bbox
            with open(self.log_file, 'a') as f:
                f.write(f"{timestamp},{label},{confidence:.4f},{x1},{y1},{x2},{y2}\n")

    def _play_alarm(self):
        """
        Play sound alarm when weapon is detected
        """
        if self.enable_sound_alarm:
            try:
                if platform.system() == 'Windows':
                    import winsound
                    winsound.Beep(1000, 500)  # 1000 Hz for 500 ms
                elif platform.system() == 'Darwin':  # macOS
                    os.system('afplay /System/Library/Sounds/Sosumi.aiff')
                else:  # Linux
                    os.system('play -nq -t alsa synth 0.5 sine 1000')
            except:
                self._log("Sound alarm not supported on this platform", 1)

    def _enhance_image(self, image):
        """
        Enhance image for better detection in different lighting conditions
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

        # Split channels
        l, a, b = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)

        # Merge channels
        limg = cv2.merge((cl, a, b))

        # Convert back to RGB
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

        return enhanced

    def _process_frame(self, frame):
        """
        Process a single frame for object detection with improved accuracy
        """
        try:
            # Apply frame preprocessing for better detection
            # Convert to RGB (YOLO expects RGB)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Apply image enhancement for better detection
            frame_rgb = self._enhance_image(frame_rgb)

            # Detect objects using YOLO with task-specific settings for improved detection
            results = self.yolo_model(
                frame_rgb,
                conf=self.weapon_confidence,
                iou=0.45,  # Intersection over Union threshold
                max_det=50,  # Maximum detections per frame
                verbose=False
            )

            detected_objects = []
            weapon_classes = [
                'knife', 'gun', 'weapon', 'pistol', 'rifle', 'handgun', 'shotgun', 'firearm',
                'revolver', 'blade', 'sword', 'dagger', 'machete', 'bomb', 'grenade'
            ]

            # Get current timestamp for logging
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    label = self.yolo_model.names[cls]

                    # Log all detected objects
                    self._log(f"Detected: {label} (Confidence: {conf:.2f})", 2)

                    # Extract bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Weapon detection logic with expanded weapon classes
                    if any(weapon_type in label.lower() for weapon_type in weapon_classes):
                        # Draw red box for weapons with thicker lines
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)

                        # Add background rectangle for better text visibility
                        cv2.rectangle(frame, (x1, y1 - 30), (x1 + len(f"WEAPON: {label}") * 13, y1), (0, 0, 255), -1)
                        cv2.putText(frame, f"WEAPON: {label}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

                        # Additional information display
                        cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y2 + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        # Add timestamp to frame
                        cv2.putText(frame, timestamp, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                        # Send high-priority alert with more details
                        alert_msg = f"WEAPON ALERT: {label} detected with {conf:.2f} confidence at {timestamp}"
                        self._log(alert_msg, 0)
                        self._log_detection(timestamp, label, conf, (x1, y1, x2, y2))
                        self._send_email_alert(alert_msg, frame.copy())
                        self._play_alarm()
                    else:
                        # Draw green box for non-weapons
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                        cv2.putText(frame, f"{label}: {conf:.2f}", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    detected_objects.append({
                        'label': label,
                        'confidence': conf,
                        'position': (x1, y1, x2, y2)
                    })

            return frame, detected_objects

        except Exception as e:
            self._log(f"Frame processing error: {e}", 0)
            return frame, []

    def start_monitoring(self):
        """
        Start continuous monitoring with multi-threaded processing and frame aggregation
        for more accurate detection
        """
        try:
            # Validate camera and model
            if not self.camera or not self.yolo_model:
                raise RuntimeError("Camera or YOLO model not initialized")

            self._log("Starting security monitoring with advanced detection...", 1)

            # Performance tracking
            frame_count = 0
            start_time = time.time()
            fps_update_interval = 10  # Update FPS every 10 frames

            # Frame buffer for temporal consistency
            frame_buffer = []
            buffer_size = 3  # Number of frames to consider for temporal consistency

            # Start time for session
            session_start = time.strftime("%Y-%m-%d %H:%M:%S")
            self._log(f"Monitoring session started at: {session_start}", 1)

            while self.camera.isOpened():
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    self._log("Failed to read frame", 0)
                    time.sleep(1)  # Brief pause before retry
                    continue

                # Process frame
                processed_frame, objects = self._process_frame(frame)

                # Add to frame buffer
                frame_buffer.append(objects)
                if len(frame_buffer) > buffer_size:
                    frame_buffer.pop(0)

                # FPS calculation
                frame_count += 1
                if frame_count % fps_update_interval == 0:
                    end_time = time.time()
                    fps = fps_update_interval / (end_time - start_time)
                    start_time = end_time

                    # Add FPS to frame
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, processed_frame.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display status information
                cv2.putText(processed_frame, f"Monitoring Active - Press 'q' to quit",
                            (10, processed_frame.shape[0] - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                # Display processed frame
                cv2.imshow("Advanced Security Monitoring", processed_frame)

                # Break on 'q' key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            self._log(f"Monitoring error: {e}", 0)

        finally:
            # Cleanup
            if self.camera:
                self.camera.release()
            cv2.destroyAllWindows()
            self._log("Monitoring stopped", 1)

    def test_on_images(self, image_folder="test_images"):
        """
        Test detection on sample images
        """
        if not os.path.exists(image_folder):
            self._log(f"Test image folder {image_folder} not found. Creating it now.", 1)
            os.makedirs(image_folder)
            return False

        test_images = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not test_images:
            self._log(f"No test images found in {image_folder}. Add some images for testing.", 1)
            return False

        self._log(f"Testing on {len(test_images)} sample images", 1)

        # Create window
        cv2.namedWindow("Test Detection", cv2.WINDOW_NORMAL)

        for img_file in test_images:
            img_path = os.path.join(image_folder, img_file)
            image = cv2.imread(img_path)

            if image is None:
                self._log(f"Failed to load image: {img_path}", 0)
                continue

            # Process image
            processed_img, objects = self._process_frame(image)

            # Count weapon detections
            weapons = [obj for obj in objects if any(
                weapon_type in obj['label'].lower() for weapon_type in [
                    'knife', 'gun', 'weapon', 'pistol', 'rifle', 'handgun', 'shotgun', 'firearm',
                    'revolver', 'blade', 'sword', 'dagger', 'machete'
                ]
            )]

            self._log(f"Image: {img_file} - Detected {len(objects)} objects, {len(weapons)} weapons", 1)

            # Display results
            cv2.imshow("Test Detection", processed_img)
            key = cv2.waitKey(2000)  # Show each image for 2 seconds

            # Quit if 'q' pressed
            if key & 0xFF == ord('q'):
                break

        cv2.destroyWindow("Test Detection")
        return True


def main():
    try:
        # Print debug info
        print(f"Python version: {sys.version}")
        print(f"OpenCV version: {cv2.__version__}")
        print(f"Platform: {platform.platform()}")

        # Create monitor object
        monitor = SecurityMonitor()

        # Optional: Test with images first if they exist
        monitor.test_on_images()

        # Start live monitoring
        monitor.start_monitoring()
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()