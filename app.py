from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from PIL import Image, ImageDraw, ImageFont
import io

app = Flask(__name__)
CORS(app)

@app.route('/scan-rice', methods=['POST'])
def scan_rice():
    try:
        data = request.json
        target_name = data.get('name', 'Simya').strip() or 'Simya'
        image_data = data.get('image')

        if not image_data:
            return jsonify({'status': 'error', 'message': 'No image provided.'}), 400

        # Decode base64 image string
        encoded_data = image_data.split(',')[1] if ',' in image_data else image_data
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return jsonify({'status': 'error', 'message': 'Invalid image file.'}), 400

        h, w, _ = img.shape
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        # Define rice bowl ROI ellipse (centers over the main rice mound)
        center_x = int(w * 0.46)
        center_y = int(h * 0.60)
        axis_x = int(w * 0.38)
        axis_y = int(h * 0.24)

        # High-resolution step grid over the rice bowl surface
        step_x = max(24, int(w / 22))
        step_y = max(20, int(h / 24))

        # Convert image to HSV for brightness/rice pixel verification
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Scale font size according to image dimension
        font_size = max(10, int(w / 48))

        # Load system font supporting clean rendering
        font = None
        for font_name in ["Nirmala.ttf", "malayalam.ttf", "arial.ttf", "segoeui.ttf"]:
            try:
                font = ImageFont.truetype(font_name, font_size)
                break
            except Exception:
                continue
                
        if font is None:
            font = ImageFont.load_default()

        detected_count = 0

        # Iterate over points within the rice bowl ellipse
        for y in range(center_y - axis_y, center_y + axis_y, step_y):
            for x in range(center_x - axis_x, center_x + axis_x, step_x):
                if 0 <= x < w and 0 <= y < h:
                    # Check if coordinate lies within bowl boundary
                    if ((x - center_x)**2 / axis_x**2 + (y - center_y)**2 / axis_y**2) <= 0.88:
                        
                        # Verify brightness (Saturation & Value) to target rice grains
                        s_val = hsv[y, x, 1]
                        v_val = hsv[y, x, 2]

                        # Filter for light/rice pixels (low saturation, higher brightness)
                        if s_val < 140 and v_val > 100:
                            angle = np.random.randint(-35, 35)

                            # Render the person's name on a transparent canvas
                            txt_w = int(font_size * len(target_name) * 0.85) + 10
                            txt_h = int(font_size * 2.2)
                            txt_img = Image.new('RGBA', (max(txt_w, 20), max(txt_h, 20)), (0, 0, 0, 0))
                            txt_draw = ImageDraw.Draw(txt_img)

                            # Custom dark brown color (#3A2012)
                            txt_draw.text((2, 2), target_name, fill=(58, 32, 18, 240), font=font)

                            # Rotate label naturally along rice grain angles
                            rotated_txt = txt_img.rotate(-angle, expand=True, resample=Image.BICUBIC)
                            
                            paste_x = int(x - rotated_txt.width / 2)
                            paste_y = int(y - rotated_txt.height / 2)

                            if 0 <= paste_x < w - rotated_txt.width and 0 <= paste_y < h - rotated_txt.height:
                                pil_img.paste(rotated_txt, (paste_x, paste_y), rotated_txt)
                                detected_count += 1

        # Encode resulting image to base64 JPEG
        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=95)
        processed_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Formulate exact requested custom message
        custom_message = f"Found {detected_count} grains are not reserved for {target_name}"

        return jsonify({
            'status': 'success',
            'count': detected_count,
            'message': custom_message,
            'image': f"data:image/jpeg;base64,{processed_base64}"
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)