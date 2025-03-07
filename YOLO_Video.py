from datetime import datetime
from ultralytics import YOLO
import cv2
import math
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Email sender details
SENDER_EMAIL = "brunoblay200f0@gmail.com"
SENDER_PASSWORD = "sjjr riiq midy lrid"
RECEIVER_EMAIL = "brunoblay2002@gmail.com"

def send_alert(detections):
    subject = "⚠️ PPE Alert: Safety Violation Detected!"
    body = "The following PPE violations were detected:\n\n"
    for detection in detections:
        body += f"- {detection['time']}: {detection['class']} (Confidence: {detection['confidence']})\n"
    
    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        server.quit()
        print("🚨 Email alert sent successfully!")
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def video_detection(path_x):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)
    
    model = YOLO("YOLO-Weights/bestest.pt")
    classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-hardhat',
                  'NO-Mask', 'NO-Safety Vest', 'Person', 'SUV', 'Safety Cone', 'Safety Vest',
                  'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 'sedan', 'semi',
                  'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

    start_time = datetime.now()
    detection_results = []

    while True:
        success, img = cap.read()
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = math.ceil((box.conf[0] * 100)) / 100
                cls = int(box.cls[0])
                class_name = classNames[cls]
                label = f'{class_name} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3

                if class_name in ['Hardhat', 'Gloves', 'Mask', 'Safety Vest']:
                    color = (0, 255, 0)  # Green for correct PPE
                elif class_name in ['NO-hardhat', 'NO-Mask', 'NO-Safety Vest']:
                    color = (0, 0, 255)  # Red for missing PPE
                else:
                    color = (85, 45, 255)

                if conf > 0.6:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                    cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
                    cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)
                    
                    # If PPE is missing, save the detection result
                    if class_name in ['NO-Mask', 'NO-Safety Vest', 'NO-hardhat']:
                        detection_results.append({
                            'class': class_name,
                            'confidence': conf,
                            'bounding_box': (x1, y1, x2, y2),
                            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        })

        yield img

        if (datetime.now() - start_time).seconds >= 30:
            if detection_results:  # Only send an email if violations were detected
                send_alert(detection_results)
            
            with open('detection_results.txt', 'a') as file:
                for detection in detection_results:
                    file.write(f"[{detection['time']}] {detection['class']} {detection['confidence']} {detection['bounding_box']}\n")
                file.write('\n')

            start_time = datetime.now()
            detection_results = []

        yield img

    cv2.destroyAllWindows()
